/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_rerank.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"
#include "lib/utility/utility.h"
#include "common/json_type/ob_json_common.h"
#include <algorithm>
 
using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{

ObExprAIRerank::ObExprAIRerank(common::ObIAllocator &alloc)
: ObFuncExprOperator(alloc, 
                    T_FUN_SYS_AI_RERANK, 
                    N_AI_RERANK, 
                    MORE_THAN_ZERO,
                    NOT_VALID_FOR_GENERATED_COL, 
                    NOT_ROW_DIMENSION) 
{
}

ObExprAIRerank::~ObExprAIRerank() 
{
}

int ObExprAIRerank::calc_result_typeN(ObExprResType &type,
                                    ObExprResType *types_stack,
                                    int64_t param_num,
                                    common::ObExprTypeCtx &type_ctx) const {
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num < 3 || param_num > 4)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    if (!ob_is_string_tc(types_stack[MODEL_IDX].get_type())) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, model key must be string type");
    } else if (!ob_is_string_type(types_stack[QUERY_IDX].get_type())) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, query must be string type");
    } else {
      types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      types_stack[QUERY_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    }
    ObObjType in_type = types_stack[DOCUMENTS_IDX].get_type();
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types_stack, DOCUMENTS_IDX, N_AI_RERANK))) {
    } else if (ob_is_string_type(in_type) && types_stack[DOCUMENTS_IDX].get_collation_type() != CS_TYPE_BINARY) {
      if (types_stack[DOCUMENTS_IDX].get_charset_type() != CHARSET_UTF8MB4) {
        types_stack[DOCUMENTS_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      }
    }
    if (OB_FAIL(ret)) {
    } else if (param_num == 4) {
      ObObjType doc_key_type = types_stack[DOC_KEY_IDX].get_type();
      if (doc_key_type == ObNullType) {
      } else if (ob_is_string_type(doc_key_type)) {
        types_stack[DOC_KEY_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      } else {  
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, doc key must be string type");
      }
    }

    if (OB_SUCC(ret)) {
      type.set_json();
      type.set_length((ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType]).get_length());
    }
  }
  return ret;
}

int ObExprAIRerank::eval_ai_rerank(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res)
{
  int ret = OB_SUCCESS;
  ObDatum *arg_model = nullptr;
  ObDatum *arg_query = nullptr;
  ObDatum *arg_documents = nullptr;
  ObDatum *arg_key = nullptr;
  if (OB_FAIL(expr.eval_param_value(ctx, arg_model, arg_query, arg_documents, arg_key))) {
  } else if (arg_model->is_null() || arg_query->is_null() || arg_documents->is_null()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObEvalCtx::TempAllocGuard guard(ctx);
    MultimodeAlloctor allocator(guard.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_RERANK));
    ObString model_id = arg_model->get_string();
    ObString query;
    ObIJsonBase *base = nullptr;
    bool is_null = false;
    const bool objects = OB_NOT_NULL(arg_key) && !arg_key->is_null();
    ObJsonArray *documents = nullptr;
    ObJsonArray *texts = nullptr;
    ObJsonArray *results = nullptr;
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo endpoint;
    query::ObIAiEndpointResolver *resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                   ctx.exec_ctx_, allocator, *arg_query, expr.args_[QUERY_IDX]->datum_meta_,
                   expr.args_[QUERY_IDX]->obj_meta_.has_lob_header(), query))) {
    } else if (model_id.empty() || query.empty()) {
      ret = OB_INVALID_ARGUMENT;
    } else if (OB_FAIL(ObJsonExprHelper::get_json_doc(
                   expr, ctx, allocator, DOCUMENTS_IDX, base, is_null, true))) {
    } else if (is_null || OB_ISNULL(base) || base->json_type() != ObJsonNodeType::J_ARRAY
               || base->element_count() == 0) {
      ret = OB_INVALID_ARGUMENT;
    } else {
      documents = static_cast<ObJsonArray *>(base);
      if (objects) {
        ObString key;
        if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                       ctx.exec_ctx_, allocator, *arg_key, expr.args_[DOC_KEY_IDX]->datum_meta_,
                       expr.args_[DOC_KEY_IDX]->obj_meta_.has_lob_header(), key))) {
        } else if (OB_FAIL(get_doc_array_from_documents_array_with_key(
                       allocator, documents, key, texts))) {
        }
      } else if (!ObAIFuncJsonUtils::ob_is_json_array_all_str(documents)) {
        ret = OB_INVALID_ARGUMENT;
      } else {
        texts = documents;
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(allocator, model_id, info))) {
    } else if (OB_ISNULL(resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(resolver->resolve_by_model_name(model_id, allocator, endpoint))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, results))) {
    } else {
      ObAIFuncModel model(allocator, *info, endpoint);
      const int64_t batch_size = 20;
      ObString score_key(SCORE_KEY);
      for (int64_t offset = 0; OB_SUCC(ret) && offset < texts->element_count(); offset += batch_size) {
        const int64_t end = std::min<int64_t>(offset + batch_size, texts->element_count());
        ObJsonArray *batch = nullptr;
        ObJsonArray *batch_results = nullptr;
        ObJsonArray *merged = nullptr;
        if (OB_FAIL(construct_batch_document_array(allocator, texts, offset, end, batch))) {
        } else if (OB_FAIL(model.call_rerank(query, batch, batch_results))) {
        } else if (OB_FAIL(batch_result_add_base(allocator, batch_results, offset))) {
        } else if (OB_FAIL(compact_json_array_by_key(
                       allocator, results, batch_results, score_key, merged))) {
        } else {
          results = merged;
        }
      }
      if (OB_SUCC(ret) && objects) {
        ObJsonArray *mapped = nullptr;
        if (OB_FAIL(sort_document_array_by_model_result(allocator, documents, results, mapped))) {
        } else {
          results = mapped;
        }
      }
      if (OB_SUCC(ret)) {
        ObString raw;
        if (OB_FAIL(results->get_raw_binary(raw, &allocator))) {
        } else if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, res, raw))) {
        }
      }
    }
  }
  return ret;
}



int ObExprAIRerank::get_doc_array_from_documents_array_with_key(ObIAllocator &allocator, ObJsonArray *document_object_array, ObString &doc_key, ObJsonArray *&doc_array)
{
  INIT_SUCC(ret);
  if (OB_ISNULL(document_object_array)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (doc_key.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonArray *doc_array_obj = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, doc_array_obj))) {
    } else {
      int64_t count = document_object_array->element_count();
      ObJsonNode *doc_node = nullptr;
      ObJsonNode *doc_value_node = nullptr;
      ObJsonObject *doc_obj = nullptr;
      for (int64_t i = 0; OB_SUCC(ret) && i < count; i++) {
        doc_node = document_object_array->get_value(i);
        if (OB_ISNULL(doc_node)) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, document_array element is null");
        } else if (doc_node->json_type() != ObJsonNodeType::J_OBJECT) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, document_array element is not object");
        } else {
          doc_obj = static_cast<ObJsonObject *>(doc_node);
          doc_value_node = doc_obj->get_value(doc_key);
          if (OB_ISNULL(doc_value_node)) {
            ret = OB_INVALID_ARGUMENT;
            LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, the value for given doc key is null");
          } else if (doc_value_node->json_type() != ObJsonNodeType::J_STRING) {
            ret = OB_INVALID_ARGUMENT;
            LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, the value for given doc key is not string");
          } else if (OB_FAIL(doc_array_obj->append(doc_value_node))) {
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      doc_array = doc_array_obj;
    }
  }
  return ret;
}

int ObExprAIRerank::sort_document_array_by_model_result(ObIAllocator &allocator, ObJsonArray *document_array, ObJsonArray *model_result_array, ObJsonArray *&sorted_document_array)
{
  INIT_SUCC(ret);
  ObJsonArray *sorted_document_array_obj = nullptr;
  if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, sorted_document_array_obj))) {
  } else {
    int64_t count = model_result_array->element_count();
    for (int64_t i = 0; OB_SUCC(ret) && i < count; i++) {
      ObJsonObject *model_result_obj = static_cast<ObJsonObject *>(model_result_array->get_value(i));
      if (OB_ISNULL(model_result_obj)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        ObJsonInt *index_value = static_cast<ObJsonInt *>(model_result_obj->get_value(INDEX_KEY));
        ObJsonDouble *score_value = static_cast<ObJsonDouble *>(model_result_obj->get_value(SCORE_KEY));
        if (OB_ISNULL(index_value) || OB_ISNULL(score_value)) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          int64_t index = index_value->value();
          ObJsonObject *origin_obj = static_cast<ObJsonObject *>(document_array->get_value(index));
          ObJsonObject *new_obj = nullptr;
          if (OB_ISNULL(origin_obj)) {
            ret = OB_ERR_UNEXPECTED;
          } else if (OB_ISNULL(new_obj = static_cast<ObJsonObject *>(origin_obj->clone(&allocator, true)))) {
            ret = OB_ERR_UNEXPECTED;
          } else if (OB_FAIL(new_obj->add(MODEL_SCORE_KEY, score_value))) {
          } else if (OB_FAIL(sorted_document_array_obj->append(new_obj))) {
          }
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    sorted_document_array = sorted_document_array_obj;
  }
  return ret;
}

int ObExprAIRerank::construct_batch_document_array(ObIAllocator &allocator, ObJsonArray *document_array, int64_t start_idx, int64_t end_idx, ObJsonArray *&batch_document_array)
{
  INIT_SUCC(ret);
  ObJsonArray *result_array = nullptr;
  if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, result_array))) {
  } else {
    for (int64_t i = start_idx; OB_SUCC(ret) && i < end_idx; i++) {
      if (OB_FAIL(result_array->append(document_array->get_value(i)))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    batch_document_array = result_array;
  }
  return ret;
}



int ObExprAIRerank::batch_result_add_base(ObIAllocator &allocator, ObJsonArray *array, int64_t start_idx)
{
  INIT_SUCC(ret);
  int64_t count = array->element_count();
  ObString index_key(INDEX_KEY);
  for (int64_t i = 0; OB_SUCC(ret) && i < count; i++) {
    ObJsonObject *obj = static_cast<ObJsonObject *>(array->get_value(i));
    if (OB_ISNULL(obj)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      ObJsonInt *int_obj = static_cast<ObJsonInt *>(obj->get_value(index_key));
      if (OB_ISNULL(int_obj)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        int64_t id = int_obj->value() + start_idx;
        int_obj->set_value(id);
      }
    }
  }
  return ret;
}

int ObExprAIRerank::compact_json_array_by_key(ObIAllocator &allocator, ObJsonArray *array1, ObJsonArray *array2, ObString &score_key, ObJsonArray *&compact_array)
{
  INIT_SUCC(ret);
  ObJsonArray *result_array = nullptr;
  if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, result_array))) {
  } else {
    int64_t count1 = array1->element_count();
    int64_t count2 = array2->element_count();
    int64_t i = 0;
    int64_t j = 0;
    while (OB_SUCC(ret) && i < count1 && j < count2) {
      ObJsonObject *obj1 = static_cast<ObJsonObject *>(array1->get_value(i));
      ObJsonObject *obj2 = static_cast<ObJsonObject *>(array2->get_value(j));
      if (OB_ISNULL(obj1) || OB_ISNULL(obj2)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        ObJsonDouble *double1 = static_cast<ObJsonDouble *>(obj1->get_value(score_key));
        ObJsonDouble *double2 = static_cast<ObJsonDouble *>(obj2->get_value(score_key));
        if (OB_ISNULL(double1) || OB_ISNULL(double2)) {
          ret = OB_ERR_UNEXPECTED;
        } else {
          if (double1->value() >= double2->value()) {
            if (OB_FAIL(result_array->append(obj1))) {
            }
            i++;
          } else {
            if (OB_FAIL(result_array->append(obj2))) {
            }
            j++;
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      while (OB_SUCC(ret) && i < count1) {
        if (OB_FAIL(result_array->append(array1->get_value(i)))) {
        }
        i++;
      }
    }
    if (OB_SUCC(ret)) {
      while (OB_SUCC(ret) && j < count2) {
        if (OB_FAIL(result_array->append(array2->get_value(j)))) {
        }
        j++;
      }
    }
  }
  if (OB_SUCC(ret)) {
    compact_array = result_array;
  }
  return ret;
}



int ObExprAIRerank::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                        const ObRawExpr &raw_expr,
                        ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   bool got_data = false;
  //   ObObj const_data;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else {
  //     ObAIFuncExprInfo *info = OB_NEWx(ObAIFuncExprInfo, allocator, *allocator, T_FUN_SYS_AI_RERANK);
  //     if (OB_ISNULL(info)) {
  //       ret = OB_ALLOCATE_MEMORY_FAILED;
  //       LOG_WARN("failed to allocate memory for ai rerank info", K(ret));
  //     } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //       LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //     } else {
  //       rt_expr.extra_info_ = info;
  //     }
  //   }
  // }
  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIRerank::eval_ai_rerank;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
