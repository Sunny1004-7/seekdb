#define USING_LOG_PREFIX SERVER

#include "ob_ai_column_worker.h"

#include <cstring>
#include <cctype>
#include "lib/oblog/ob_log.h"
#include "lib/allocator/page_arena.h"
#include "lib/container/ob_array.h"
#include "common/mysqlclient/ob_mysql_proxy.h"
#include "common/mysqlclient/ob_mysql_result.h"
#include "share/ob_server_struct.h"
#include "share/ob_server_status.h"
#include "share/rc/ob_server_runtime.h"
#include "share/ai_service/ob_ai_service_struct.h"
#include "query/api/query/ai/ob_ai_endpoint_resolver.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_client.h"

namespace oceanbase
{
using namespace common;
using namespace share;
namespace observer
{

namespace ai_column_worker {

const int64_t TASK_STATE_PENDING = 1;
const int64_t TASK_STATE_RETRYING = 3;
const int64_t TASK_BATCH_SIZE = 32;
const int64_t WORKER_IDLE_US = 50 * 1000;

const char *const AI_JSON_TYPE_PROMPT = "Return exactly one valid JSON value.";
const char *const AI_TEXT_TYPE_PROMPT =
    "Return the answer as plain text without extra explanation.";
const char *const AI_INT_TYPE_PROMPT =
    "Return the answer as a base-10 integer without extra explanation.";

int append_sql_string(ObSqlString &sql, const ObString &value)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(sql.append("'"))) {
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < value.length(); ++i) {
    const char c = value.ptr()[i];
    if ('\'' == c) {
      ret = sql.append("''");
    } else if ('\\' == c) {
      ret = sql.append("\\\\");
    } else {
      ret = sql.append(&c, 1);
    }
  }
  if (OB_SUCC(ret)) {
    ret = sql.append("'");
  }
  return ret;
}

// AI_COMPLETE('model', <prompt_expr>[, config]) -> <prompt_expr>
int extract_prompt_expr(const ObString &expr, ObString &prompt_expr)
{
  int ret = OB_SUCCESS;
  int64_t open_pos = -1;
  for (int64_t i = 0; OB_SUCC(ret) && i < expr.length(); ++i) {
    if ('(' == expr.ptr()[i]) {
      open_pos = i;
      break;
    }
  }
  if (open_pos < 0) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    int64_t depth = 0;
    int64_t arg_index = 0;
    int64_t arg_start = open_pos + 1;
    bool in_single_quote = false;
    bool in_double_quote = false;
    bool in_back_quote = false;
    bool finished = false;
    for (int64_t i = open_pos + 1; OB_SUCC(ret) && !finished && i < expr.length(); ++i) {
      const char c = expr.ptr()[i];
      if (in_single_quote) {
        if ('\\' == c) {
          ++i;
        } else if ('\'' == c) {
          in_single_quote = false;
        }
      } else if (in_double_quote) {
        if ('\\' == c) {
          ++i;
        } else if ('"' == c) {
          in_double_quote = false;
        }
      } else if (in_back_quote) {
        if ('`' == c) {
          in_back_quote = false;
        }
      } else if ('\'' == c) {
        in_single_quote = true;
      } else if ('"' == c) {
        in_double_quote = true;
      } else if ('`' == c) {
        in_back_quote = true;
      } else if ('(' == c) {
        ++depth;
      } else if (')' == c) {
        if (0 == depth) {
          if (1 == arg_index) {
            int64_t end = i;
            while (end > arg_start && isspace(static_cast<unsigned char>(expr.ptr()[end - 1]))) {
              --end;
            }
            while (arg_start < end && isspace(static_cast<unsigned char>(expr.ptr()[arg_start]))) {
              ++arg_start;
            }
            prompt_expr.assign_ptr(expr.ptr() + arg_start, static_cast<int32_t>(end - arg_start));
          } else {
            ret = OB_INVALID_ARGUMENT;
          }
          finished = true;
        } else {
          --depth;
        }
      } else if (',' == c && 0 == depth) {
        ++arg_index;
        arg_start = i + 1;
      }
    }
    if (OB_SUCC(ret) && !finished) {
      ret = OB_INVALID_ARGUMENT;
    }
  }
  return ret;
}

void pick_system_prompt(const ObString &normalized_expr, ObString &system_prompt)
{
  const int64_t json_len = static_cast<int64_t>(strlen(AI_JSON_TYPE_PROMPT));
  const int64_t text_len = static_cast<int64_t>(strlen(AI_TEXT_TYPE_PROMPT));
  if (normalized_expr.length() >= json_len
      && 0 == strncmp(normalized_expr.ptr() + normalized_expr.length() - json_len,
                      AI_JSON_TYPE_PROMPT, json_len)) {
    system_prompt.assign_ptr(AI_JSON_TYPE_PROMPT, static_cast<int32_t>(json_len));
  } else if (normalized_expr.length() >= text_len
             && 0 == strncmp(normalized_expr.ptr() + normalized_expr.length() - text_len,
                             AI_TEXT_TYPE_PROMPT, text_len)) {
    system_prompt.assign_ptr(AI_TEXT_TYPE_PROMPT, static_cast<int32_t>(text_len));
  } else {
    const int64_t int_len = static_cast<int64_t>(strlen(AI_INT_TYPE_PROMPT));
    if (normalized_expr.length() >= int_len
        && 0 == strncmp(normalized_expr.ptr() + normalized_expr.length() - int_len,
                        AI_INT_TYPE_PROMPT, int_len)) {
      system_prompt.assign_ptr(AI_INT_TYPE_PROMPT, static_cast<int32_t>(int_len));
    }
  }
}

} // namespace ai_column_worker

using namespace ai_column_worker;

int ObAiColumnTaskWorker::server_module_init(ObAiColumnTaskWorker *&worker)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(worker)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(worker->init())) {
  }
  return ret;
}

int ObAiColumnTaskWorker::init()
{
  int ret = OB_SUCCESS;
  if (inited_) {
    ret = OB_INIT_TWICE;
  } else if (OB_FAIL(ObServerThreadHelper::create("AiColWorker", 1))) {
  } else {
    inited_ = true;
    LOG_INFO("[AI_COLUMN] worker init success");
  }
  return ret;
}

int ObAiColumnTaskWorker::start()
{
  int ret = OB_SUCCESS;
  if (!inited_) {
    ret = OB_NOT_INIT;
  } else if (OB_FAIL(ObServerThreadHelper::start())) {
  }
  return ret;
}

void ObAiColumnTaskWorker::stop()
{
  ObServerThreadHelper::stop();
}

void ObAiColumnTaskWorker::wait()
{
  ObServerThreadHelper::wait();
}

void ObAiColumnTaskWorker::destroy()
{
  if (inited_) {
    ObServerThreadHelper::destroy();
    inited_ = false;
  }
}

void ObAiColumnTaskWorker::do_work()
{
  int ret = OB_SUCCESS;
  LOG_INFO("[AI_COLUMN] worker thread started");
  while (!has_set_stop()) {
    // the observer serves SQL only after bootstrap; never touch inner tables before it
    if (share::SS_SERVING == GCTX.status_) {
      if (OB_FAIL(process_tasks())) {
        LOG_ERROR("[AI_COLUMN] process tasks failed", K(ret));
      }
    }
    idle(WORKER_IDLE_US);
  }
  LOG_INFO("[AI_COLUMN] worker thread stopped");
}

int ObAiColumnTaskWorker::process_tasks()
{
  int ret = OB_SUCCESS;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  ObSqlString sql;
  ObArenaAllocator task_alloc("AiColTasks");
  ObSEArray<AiColumnTaskRow, 8> tasks;
  if (OB_ISNULL(proxy)) {
    // sql proxy is not ready yet, try again later
  } else if (OB_FAIL(sql.append_fmt(
                 "select t.task_id, t.table_id, t.column_id, t.request_id, t.retry_count, "
                 "t.rowkey_predicate, a.column_name, a.model_name, a.ai_column_expr, "
                 "a.normalized_expr, a.max_retries, tb.table_name, db.database_name "
                 "from oceanbase.__all_ai_column_task t "
                 "join oceanbase.__all_ai_column a on a.table_id = t.table_id "
                 "and a.column_id = t.column_id "
                 "join oceanbase.__all_table tb on tb.table_id = t.table_id "
                 "join oceanbase.__all_database db on db.database_id = tb.database_id "
                 "where t.state in (%ld, %ld) "
                 "and (t.eligible_ts is null or t.eligible_ts <= now(6)) "
                 "order by t.task_id limit %ld",
                 TASK_STATE_PENDING, TASK_STATE_RETRYING, TASK_BATCH_SIZE))) {
  } else {
    SMART_VAR(ObMySQLProxy::MySQLResult, res) {
      sqlclient::ObMySQLResult *result = NULL;
      if (OB_FAIL(proxy->read(res, sql.ptr()))) {
      } else if (OB_ISNULL(result = res.get_result())) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        while (OB_SUCC(ret)) {
          ret = result->next();
          if (OB_ITER_END == ret) {
            ret = OB_SUCCESS;
            break;
          } else if (OB_FAIL(ret)) {
          } else {
            int64_t task_id = 0;
            int64_t table_id = 0;
            int64_t column_id = 0;
            int64_t request_id = 0;
            int64_t retry_count = 0;
            int64_t max_retries = 0;
            ObString rowkey_predicate;
            ObString column_name;
            ObString model_name;
            ObString ai_column_expr;
            ObString normalized_expr;
            ObString table_name;
            ObString database_name;
            EXTRACT_INT_FIELD_MYSQL_SKIP_RET(*result, "task_id", task_id, int64_t);
            EXTRACT_INT_FIELD_MYSQL_SKIP_RET(*result, "table_id", table_id, int64_t);
            EXTRACT_INT_FIELD_MYSQL_SKIP_RET(*result, "column_id", column_id, int64_t);
            EXTRACT_INT_FIELD_MYSQL_SKIP_RET(*result, "request_id", request_id, int64_t);
            EXTRACT_INT_FIELD_MYSQL_SKIP_RET(*result, "retry_count", retry_count, int64_t);
            EXTRACT_INT_FIELD_MYSQL_SKIP_RET(*result, "max_retries", max_retries, int64_t);
            EXTRACT_VARCHAR_FIELD_MYSQL_SKIP_RET(*result, "rowkey_predicate", rowkey_predicate);
            EXTRACT_VARCHAR_FIELD_MYSQL_SKIP_RET(*result, "column_name", column_name);
            EXTRACT_VARCHAR_FIELD_MYSQL_SKIP_RET(*result, "model_name", model_name);
            EXTRACT_VARCHAR_FIELD_MYSQL_SKIP_RET(*result, "ai_column_expr", ai_column_expr);
            EXTRACT_VARCHAR_FIELD_MYSQL_SKIP_RET(*result, "normalized_expr", normalized_expr);
            EXTRACT_VARCHAR_FIELD_MYSQL_SKIP_RET(*result, "table_name", table_name);
            EXTRACT_VARCHAR_FIELD_MYSQL_SKIP_RET(*result, "database_name", database_name);
            if (OB_SUCC(ret)) {
              AiColumnTaskRow row;
              row.task_id_ = task_id;
              row.table_id_ = static_cast<uint64_t>(table_id);
              row.column_id_ = static_cast<uint64_t>(column_id);
              row.request_id_ = static_cast<uint64_t>(request_id);
              row.retry_count_ = retry_count;
              row.max_retries_ = max_retries;
              if (OB_FAIL(ob_write_string(task_alloc, rowkey_predicate,
                                          row.rowkey_predicate_))
                  || OB_FAIL(ob_write_string(task_alloc, column_name, row.column_name_))
                  || OB_FAIL(ob_write_string(task_alloc, model_name, row.model_name_))
                  || OB_FAIL(ob_write_string(task_alloc, ai_column_expr,
                                             row.ai_column_expr_))
                  || OB_FAIL(ob_write_string(task_alloc, normalized_expr,
                                             row.normalized_expr_))
                  || OB_FAIL(ob_write_string(task_alloc, database_name,
                                             row.database_name_))
                  || OB_FAIL(ob_write_string(task_alloc, table_name,
                                             row.table_name_))) {
              } else if (OB_FAIL(tasks.push_back(row))) {
              }
            }
          }
        }
      }
    }
  }
  // Tasks of the same row change share one request_id, publish them together.
  for (int64_t i = 0; OB_SUCC(ret) && i < tasks.count(); ++i) {
    AiColumnTaskRow &head = tasks.at(i);
    if (head.grouped_) {
      continue;
    }
    ObSEArray<AiColumnTaskRow *, 4> group;
    for (int64_t j = i; j < tasks.count(); ++j) {
      AiColumnTaskRow &row = tasks.at(j);
      if (!row.grouped_
          && row.table_id_ == head.table_id_
          && row.request_id_ == head.request_id_
          && 0 == row.rowkey_predicate_.compare(head.rowkey_predicate_)) {
        if (OB_FAIL(group.push_back(&row))) {
        } else {
          row.grouped_ = true;
        }
      }
    }
    if (OB_SUCC(ret) && !group.empty()) {
      if (OB_FAIL(process_task_group(group))) {
        LOG_ERROR("[AI_COLUMN] process task group failed", K(ret), K(group.count()));
        ret = OB_SUCCESS;
      }
    }
  }
  return ret;
}

int ObAiColumnTaskWorker::process_task_group(common::ObSEArray<AiColumnTaskRow *, 4> &group)
{
  int ret = OB_SUCCESS;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  ObArenaAllocator alloc("AiColGroup");
  ObSqlString update_sql;
  ObSqlString value_sql;
  ObSqlString task_ids;
  ObSEArray<AiColumnTaskRow *, 4> succeeded;
  if (OB_ISNULL(proxy) || group.empty()) {
    ret = OB_NOT_INIT;
  } else {
    AiColumnTaskRow *head = group.at(0);
    for (int64_t i = 0; OB_SUCC(ret) && i < group.count(); ++i) {
      AiColumnTaskRow *row = group.at(i);
      ObString prompt_expr;
      ObString system_prompt;
      ObString prompt_value;
      ObSqlString result;
      pick_system_prompt(row->normalized_expr_, system_prompt);
      if (OB_FAIL(extract_prompt_expr(row->ai_column_expr_, prompt_expr))) {
        LOG_ERROR("[AI_COLUMN] bad ai column expression", K(ret), K(row->task_id_),
                  K(row->ai_column_expr_));
      } else if (OB_FAIL(read_prompt_value(head->database_name_, head->table_name_,
                                           row->rowkey_predicate_, prompt_expr,
                                           prompt_value, alloc))) {
      } else if (prompt_value.empty()) {
        LOG_ERROR("[AI_COLUMN] empty prompt, drop task", K(row->task_id_),
                  K(row->rowkey_predicate_));
        if (OB_FAIL(remove_task(row->task_id_))) {
        }
      } else if (OB_FAIL(execute_ai_request(row->model_name_, system_prompt,
                                            prompt_value, result))) {
        LOG_ERROR("[AI_COLUMN] ai request failed", K(ret), K(row->task_id_),
                  K(row->model_name_));
        ret = OB_SUCCESS;
        if (OB_FAIL(retry_later(row->task_id_, row->retry_count_, row->max_retries_))) {
        }
      } else {
        // build the column assignment
        if (0 == value_sql.length()) {
          if (OB_FAIL(value_sql.append_fmt("`%.*s` = ",
                                           static_cast<int>(row->column_name_.length()),
                                           row->column_name_.ptr()))) {
          }
        } else if (OB_FAIL(value_sql.append_fmt(", `%.*s` = ",
                                                static_cast<int>(row->column_name_.length()),
                                                row->column_name_.ptr()))) {
        }
        if (OB_SUCC(ret) && OB_FAIL(append_sql_string(value_sql, result.string()))) {
        }
        if (OB_SUCC(ret) && OB_FAIL(succeeded.push_back(row))) {
        }
      }
    }
    if (OB_SUCC(ret) && !succeeded.empty()) {
      if (OB_FAIL(update_sql.append_fmt("update `%.*s`.`%.*s` set ",
                                        static_cast<int>(head->database_name_.length()),
                                        head->database_name_.ptr(),
                                        static_cast<int>(head->table_name_.length()),
                                        head->table_name_.ptr()))) {
      } else if (OB_FAIL(update_sql.append(value_sql.string()))) {
      } else if (OB_FAIL(update_sql.append(" where "))) {
      } else if (OB_FAIL(update_sql.append(head->rowkey_predicate_))) {
      } else {
        int64_t affected_rows = 0;
        if (OB_FAIL(proxy->write(update_sql.ptr(), affected_rows))) {
          LOG_ERROR("[AI_COLUMN] grouped write back failed", K(ret), K(update_sql));
        } else {
          for (int64_t i = 0; OB_SUCC(ret) && i < succeeded.count(); ++i) {
            if (OB_FAIL(task_ids.append_fmt("%s%ld", 0 == i ? "" : ",",
                                            succeeded.at(i)->task_id_))) {
            }
          }
          if (OB_SUCC(ret) && OB_FAIL(delete_tasks(task_ids.string()))) {
          }
        }
      }
    }
  }
  return ret;
}

int ObAiColumnTaskWorker::read_prompt_value(const ObString &database_name,
                                           const ObString &table_name,
                                           const ObString &rowkey_predicate,
                                           const ObString &prompt_expr,
                                           ObString &prompt_value,
                                           common::ObIAllocator &alloc)
{
  int ret = OB_SUCCESS;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  ObSqlString sql;
  if (OB_ISNULL(proxy)) {
    ret = OB_NOT_INIT;
  } else if (OB_FAIL(sql.append_fmt("select ( %.*s ) as ai_prompt from `%.*s`.`%.*s` where ",
                                    static_cast<int>(prompt_expr.length()), prompt_expr.ptr(),
                                    static_cast<int>(database_name.length()),
                                    database_name.ptr(),
                                    static_cast<int>(table_name.length()),
                                    table_name.ptr()))) {
  } else if (OB_FAIL(sql.append(rowkey_predicate))) {
  } else {
    SMART_VAR(ObMySQLProxy::MySQLResult, res) {
      sqlclient::ObMySQLResult *result_set = NULL;
      if (OB_FAIL(proxy->read(res, sql.ptr()))) {
      } else if (OB_ISNULL(result_set = res.get_result())) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_ITER_END == (ret = result_set->next())) {
        ret = OB_SUCCESS;
      } else if (OB_FAIL(ret)) {
      } else {
        ObString tmp_value;
        if (OB_FAIL(result_set->get_varchar(static_cast<int64_t>(0), tmp_value))) {
          LOG_ERROR("[AI_COLUMN] read prompt value failed", K(ret));
        } else if (OB_FAIL(ob_write_string(alloc, tmp_value, prompt_value))) {
        }
      }
    }
  }
  return ret;
}

int ObAiColumnTaskWorker::delete_tasks(const common::ObString &task_ids)
{
  int ret = OB_SUCCESS;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  ObSqlString sql;
  int64_t affected_rows = 0;
  if (OB_ISNULL(proxy)) {
    ret = OB_NOT_INIT;
  } else if (OB_FAIL(sql.append_fmt(
                 "delete from oceanbase.__all_ai_column_task where task_id in (%.*s)",
                 static_cast<int>(task_ids.length()), task_ids.ptr()))) {
  } else if (OB_FAIL(proxy->write(sql.ptr(), affected_rows))) {
  }
  return ret;
}

int ObAiColumnTaskWorker::execute_ai_request(const ObString &model_name,
                                             const ObString &system_prompt,
                                             const ObString &user_prompt,
                                             ObSqlString &result)
{
  int ret = OB_SUCCESS;
  ObArenaAllocator alloc("AiColReq");
  ObAIFuncExprInfo *info = NULL;
  ObAiModelEndpointInfo endpoint;
  query::ObIAiEndpointResolver *resolver =
      share::server_service<query::ObIAiEndpointResolver>();
  if (OB_ISNULL(resolver)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(alloc, model_name, info))) {
  } else if (OB_ISNULL(info)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(resolver->resolve_by_model_name(model_name, alloc, endpoint, false))) {
  } else if (share::EndpointType::DENSE_EMBEDDING == info->type_) {
    ObAIFuncModel model(alloc, *info, endpoint);
    ObString content = user_prompt;
    ObString vector_str;
    if (OB_FAIL(model.call_dense_embedding(content, NULL, vector_str))) {
      LOG_ERROR("[AI_COLUMN] embedding request failed", K(ret), K(model_name));
    } else if (OB_FAIL(result.assign(vector_str))) {
    }
  } else {
    ObArray<ObString> headers;
    ObJsonObject *body = NULL;
    ObString system_prompt_copy = system_prompt;
    ObString user_prompt_copy = user_prompt;
    if (OB_FAIL(ObAIFuncUtils::get_header(alloc, *info, endpoint, headers))) {
    } else if (OB_FAIL(ObAIFuncUtils::get_complete_body(alloc, *info, endpoint,
                                                        system_prompt_copy,
                                                        user_prompt_copy, NULL, body))) {
    } else {
      ObAIFuncClient client;
      ObJsonObject *response = NULL;
      if (OB_FAIL(client.send_post(alloc, endpoint.get_url(), headers, body, response))) {
      } else {
        ObIJsonBase *result_base = NULL;
        ObString result_str;
        if (OB_FAIL(ObAIFuncUtils::parse_complete_output(alloc, endpoint,
                                                         response, result_base))) {
        } else if (OB_FAIL(ObAIFuncJsonUtils::print_json_to_str(alloc, result_base,
                                                                result_str))) {
        } else if (OB_FAIL(result.assign(result_str))) {
        }
      }
    }
  }
  return ret;
}

int ObAiColumnTaskWorker::write_back(const ObString &database_name,
                                     const ObString &table_name,
                                     const ObString &column_name,
                                     const ObString &rowkey_predicate,
                                     const ObString &value)
{
  int ret = OB_SUCCESS;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  ObSqlString sql;
  int64_t affected_rows = 0;
  if (OB_ISNULL(proxy)) {
    ret = OB_NOT_INIT;
  } else if (OB_FAIL(sql.append_fmt("update `%.*s`.`%.*s` set `%.*s` = ",
                                    static_cast<int>(database_name.length()), database_name.ptr(),
                                    static_cast<int>(table_name.length()), table_name.ptr(),
                                    static_cast<int>(column_name.length()), column_name.ptr()))) {
  } else if (OB_FAIL(append_sql_string(sql, value))) {
  } else if (OB_FAIL(sql.append(" where "))) {
  } else if (OB_FAIL(sql.append(rowkey_predicate))) {
  } else if (OB_FAIL(proxy->write(sql.ptr(), affected_rows))) {
  }
  return ret;
}

int ObAiColumnTaskWorker::remove_task(const int64_t task_id)
{
  int ret = OB_SUCCESS;
  ObMySQLProxy *proxy = GCTX.sql_proxy_;
  ObSqlString sql;
  int64_t affected_rows = 0;
  if (OB_ISNULL(proxy)) {
    ret = OB_NOT_INIT;
  } else if (OB_FAIL(sql.append_fmt(
                 "delete from oceanbase.__all_ai_column_task where task_id = %ld", task_id))) {
  } else if (OB_FAIL(proxy->write(sql.ptr(), affected_rows))) {
  }
  return ret;
}

int ObAiColumnTaskWorker::retry_later(const int64_t task_id,
                                      const int64_t retry_count,
                                      const int64_t max_retries)
{
  int ret = OB_SUCCESS;
  if (retry_count + 1 > max_retries) {
    ret = remove_task(task_id);
  } else {
    ObMySQLProxy *proxy = GCTX.sql_proxy_;
    ObSqlString sql;
    int64_t affected_rows = 0;
    if (OB_ISNULL(proxy)) {
      ret = OB_NOT_INIT;
    } else if (OB_FAIL(sql.append_fmt(
                   "update oceanbase.__all_ai_column_task set state = %ld, retry_count = %ld, "
                   "eligible_ts = timestampadd(microsecond, 100000, now(6)) where task_id = %ld",
                   TASK_STATE_RETRYING, retry_count + 1, task_id))) {
    } else if (OB_FAIL(proxy->write(sql.ptr(), affected_rows))) {
    }
  }
  return ret;
}

} // namespace observer
} // namespace oceanbase
