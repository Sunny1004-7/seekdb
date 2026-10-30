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

#define USING_LOG_PREFIX SQL_RESV
#include "sql/resolver/ddl/ob_explain_resolver.h"
#include "sql/resolver/ob_resolver.h"
#include "sql/parser/ob_parser.h"

using namespace oceanbase;
using namespace sql;
using namespace common;
const static int64_t EXPLAIN_FORMAT = 0;
const static int64_t EXPLAIN_DISPLAY_OPTION = 1;
const static int64_t EXPLAIN_INTO_TABLE = 2;
const static int64_t EXPLAIN_STMT_ID = 3;
const static int64_t EXPLAIN_CHILD_STMT = 4;

namespace
{

const char *file_col_type_name(int64_t type)
{
  // must match filetable::FileColType
  switch (type) {
    case 0: return "BIGINT";
    case 1: return "DOUBLE";
    case 2: return "BOOLEAN";
    case 3: return "DATE";
    case 4: return "DATETIME";
    default: return "VARCHAR";
  }
}

void escape_sql_literal(const ObString &in, std::string &out)
{
  out.clear();
  for (int64_t i = 0; i < in.length(); ++i) {
    if ('\'' == in[i] || '\\' == in[i]) {
      out.push_back('\\');
    }
    out.push_back(in.ptr()[i]);
  }
}

}  // namespace

int ObExplainResolver::try_resolve_file_describe(const ParseNode &child_node,
                                                 bool &handled,
                                                 ObStmt *&new_stmt)
{
  int ret = OB_SUCCESS;
  ObStmt *child_stmt = NULL;
  handled = false;
  new_stmt = NULL;
  if (OB_ISNULL(params_.allocator_) || OB_ISNULL(params_.session_info_)) {
    ret = OB_NOT_INIT;
    LOG_WARN("resolver params not inited", K(ret));
  } else {
    ObResolver sub_resolver(params_);
    if (OB_FAIL(sub_resolver.resolve(ObResolver::IS_NOT_PREPARED_STMT,
                                     child_node, child_stmt))) {
    } else if (OB_ISNULL(child_stmt) || !child_stmt->is_select_stmt()) {
      // DESCRIBE of a non-select keeps the EXPLAIN behavior
    } else {
      ObSelectStmt *select_stmt = static_cast<ObSelectStmt *>(child_stmt);
      const ObJsonTableDef *file_def = NULL;
      ObIArray<TableItem *> &table_items = select_stmt->get_table_items();
      for (int64_t i = 0; OB_ISNULL(file_def) && i < table_items.count(); ++i) {
        TableItem *item = table_items.at(i);
        if (OB_NOT_NULL(item) && item->is_json_table() && OB_NOT_NULL(item->json_table_def_)
            && !item->json_table_def_->file_cols_.empty()) {
          file_def = item->json_table_def_;
        }
      }
      if (OB_NOT_NULL(file_def) && !select_stmt->get_select_items().empty()) {
        ObSqlString sql;
        const ObIArray<SelectItem> &select_items = select_stmt->get_select_items();
        for (int64_t i = 0; OB_SUCC(ret) && i < select_items.count(); ++i) {
          const SelectItem &item = select_items.at(i);
          ObString name = item.alias_name_.empty() ? item.expr_name_ : item.alias_name_;
          const char *type_name = "VARCHAR";
          bool nullable = true;
          for (int64_t c = 0; c < file_def->file_cols_.count(); ++c) {
            const ObFileTableColumnMeta &col = file_def->file_cols_.at(c);
            if (0 == name.case_compare(col.name_)) {
              type_name = file_col_type_name(col.type_);
              nullable = col.nullable_;
              break;
            }
          }
          std::string escaped;
          escape_sql_literal(name, escaped);
          if (0 == i) {
            if (OB_FAIL(sql.append_fmt("select '%.*s' as column_name, '%s' as column_type,"
                                       " '%s' as `null`, cast(NULL as char) as `key`,"
                                       " cast(NULL as char) as `default`,"
                                       " cast(NULL as char) as extra",
                                       static_cast<int32_t>(escaped.size()), escaped.c_str(),
                                       type_name, nullable ? "YES" : "NO"))) {
              LOG_WARN("failed to build describe sql", K(ret));
            }
          } else if (OB_FAIL(sql.append_fmt(" union all select '%.*s', '%s', '%s',"
                                            " cast(NULL as char), cast(NULL as char),"
                                            " cast(NULL as char)",
                                            static_cast<int32_t>(escaped.size()), escaped.c_str(),
                                            type_name, nullable ? "YES" : "NO"))) {
            LOG_WARN("failed to build describe sql", K(ret));
          }
        }
        ParseResult parse_result;
        ObParser parser(*params_.allocator_, params_.session_info_->get_sql_mode(),
                        params_.session_info_->get_charsets4parser());
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(parser.parse(sql.string(), parse_result))) {
          LOG_WARN("failed to parse describe sql", K(ret), K(sql));
        } else if (OB_ISNULL(parse_result.result_tree_)
                   || 1 != parse_result.result_tree_->num_child_
                   || OB_ISNULL(parse_result.result_tree_->children_[0])) {
          ret = OB_ERR_UNEXPECTED;
          LOG_WARN("unexpected describe parse tree", K(ret));
        } else {
          ObResolver describe_resolver(params_);
          if (OB_FAIL(describe_resolver.resolve(ObResolver::IS_NOT_PREPARED_STMT,
                                                *parse_result.result_tree_->children_[0],
                                                new_stmt))) {
          } else if (OB_ISNULL(new_stmt)) {
            ret = OB_ERR_UNEXPECTED;
          } else {
            handled = true;
          }
        }
      }
    }
  }
  return ret;
}

int ObExplainResolver::resolve(const ParseNode &parse_tree)
{
  int ret = OB_SUCCESS;
  ObExplainStmt *explain_stmt = NULL;
  if (!(T_EXPLAIN == parse_tree.type_  && 5 == parse_tree.num_child_)) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("invalid EXPLAIN syntax",
             "type", parse_tree.type_,
             "# of child", parse_tree.num_child_);
  } else if (NULL == (explain_stmt = create_stmt<ObExplainStmt>())) {
    ret = OB_SQL_RESOLVER_NO_MEMORY;
    LOG_WARN("failed to create explain stmt");
  } else {
    ParseNode *child_node = parse_tree.children_[EXPLAIN_CHILD_STMT];
    ParseNode *format_node = parse_tree.children_[EXPLAIN_FORMAT];
    const bool is_describe = (OB_NOT_NULL(format_node) && T_DESCRIBE == format_node->type_);
    bool described = false;
    ObStmt *describe_stmt = NULL;
    if (is_describe && OB_NOT_NULL(child_node) && T_SELECT == child_node->type_) {
      if (OB_FAIL(try_resolve_file_describe(*child_node, described, describe_stmt))) {
      } else if (described) {
        set_basic_stmt(describe_stmt);
        return ret;
      }
    }
    ObStmt *child_stmt = NULL;
    ObExplainDisplayOpt opt;
    opt.with_tree_line_ = true;
    if (OB_ISNULL(child_node)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("empty parse tree for stmt");
    } else if (!(T_SELECT == child_node->type_
                 || T_DELETE == child_node->type_
                 || T_INSERT == child_node->type_
                 || T_UPDATE == child_node->type_
                 || T_REPLACE == child_node->type_)) {
      ret = OB_NOT_SUPPORTED;
      LOG_WARN("invalid stmt type for EXPLAIN", "type", child_node->type_);
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "stmt type for EXPLAIN");
    } else {
      ObResolver resolver(params_);
      if (OB_FAIL(resolver.resolve(ObResolver::IS_NOT_PREPARED_STMT,
                                   *child_node,
                                   child_stmt))) {
      } else if (OB_ISNULL(child_stmt)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("failed to generate stmt for child stmt");
      } else {
        explain_stmt->set_explain_query_stmt(static_cast<ObDMLStmt *>(child_stmt));
        ParseNode *format = parse_tree.children_[EXPLAIN_FORMAT];
        if (NULL != format && !is_describe) {
          switch (format->type_) {
          case T_OUTLINE:
          {
            explain_stmt->set_explain_format(EXPLAIN_OUTLINE);
            opt.with_tree_line_ = true;
          } break;
          case T_EXTENDED:
          {
            explain_stmt->set_explain_format(EXPLAIN_EXTENDED);
            opt.with_tree_line_ = true;
          } break;
          case T_PARTITIONS:
          {
            explain_stmt->set_explain_format(EXPLAIN_PARTITIONS);
            opt.with_tree_line_ = true;
          } break;
          case T_TRADITIONAL:
          {
            explain_stmt->set_explain_format(EXPLAIN_TRADITIONAL);
            opt.with_tree_line_ = true;
          } break;
          case T_FORMAT_JSON:
          {
            explain_stmt->set_explain_format(EXPLAIN_FORMAT_JSON);
            opt.with_tree_line_ = false;
          } break;
          case T_BASIC: {
            explain_stmt->set_explain_format(EXPLAIN_BASIC);
            opt.with_tree_line_ = true;
            break;
          }
          case T_EXTENDED_NOADDR: {
            explain_stmt->set_explain_format(EXPLAIN_EXTENDED_NOADDR);
            opt.with_tree_line_ = true;
            break;
          }
          case T_PLANREGRESS: {
            explain_stmt->set_explain_format(EXPLAIN_PLANREGRESS);
            opt.with_tree_line_ = true;
            break;
          }
          default:
            ret = OB_ERROR;
            LOG_WARN("unknown explain format", "format", format->type_);
            break;
          }
        }
      }
    }
    ParseNode *opt_node = parse_tree.children_[EXPLAIN_DISPLAY_OPTION];
    if (OB_SUCC(ret) && NULL != opt_node) {
      switch (opt_node->type_) {
      case T_PRETTY_COLOR:
        opt.with_color_ = true;
        // pass
      case T_PRETTY:
        opt.with_tree_line_ = true;
        break;
      default:
        ret = OB_ERR_UNEXPECTED;
      }
    }
    if (OB_SUCC(ret)) {
      explain_stmt->set_display_opt(opt);
    }
    ParseNode *into_table_node = parse_tree.children_[EXPLAIN_INTO_TABLE];
    if (OB_SUCC(ret) && NULL != into_table_node) {
      ObString into_table(into_table_node->str_len_, into_table_node->str_value_);
      explain_stmt->set_into_table(into_table);
    }
    ParseNode *statement_id_node = parse_tree.children_[EXPLAIN_STMT_ID];
    if (OB_SUCC(ret) && NULL != statement_id_node) {
      ObString statement_id(statement_id_node->str_len_, statement_id_node->str_value_);
      explain_stmt->set_statement_id(statement_id);
    }
  }

  return ret;
}
