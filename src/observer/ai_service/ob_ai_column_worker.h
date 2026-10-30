#ifndef OCEANBASE_OBSERVER_AI_SERVICE_OB_AI_COLUMN_WORKER_H_
#define OCEANBASE_OBSERVER_AI_SERVICE_OB_AI_COLUMN_WORKER_H_

#include "rootserver/ob_server_thread_helper.h"
#include "lib/container/ob_se_array.h"
#include "lib/string/ob_sql_string.h"

namespace oceanbase
{
namespace observer
{

struct AiColumnTaskRow
{
  AiColumnTaskRow()
    : task_id_(0), table_id_(0), column_id_(0), request_id_(0), retry_count_(0),
      max_retries_(0), rowkey_predicate_(), column_name_(), model_name_(),
      ai_column_expr_(), normalized_expr_(), database_name_(), table_name_(),
      grouped_(false)
  {}
  int64_t task_id_;
  uint64_t table_id_;
  uint64_t column_id_;
  uint64_t request_id_;
  int64_t retry_count_;
  int64_t max_retries_;
  common::ObString rowkey_predicate_;
  common::ObString column_name_;
  common::ObString model_name_;
  common::ObString ai_column_expr_;
  common::ObString normalized_expr_;
  common::ObString database_name_;
  common::ObString table_name_;
  bool grouped_;
};

class ObAiColumnTaskWorker : public rootserver::ObServerThreadHelper
{
public:
  ObAiColumnTaskWorker() : inited_(false) {}
  virtual ~ObAiColumnTaskWorker() { destroy(); }
  static int server_module_init(ObAiColumnTaskWorker *&worker);
  int init();
  int start();
  void stop() override;
  void wait() override;
  void destroy();
  void do_work() override;
  void deactivate() override {}
  int activate() override { return common::OB_SUCCESS; }

private:
  int process_tasks();
  int process_task_group(common::ObSEArray<AiColumnTaskRow *, 4> &group);
  int read_prompt_value(const common::ObString &database_name,
                        const common::ObString &table_name,
                        const common::ObString &rowkey_predicate,
                        const common::ObString &prompt_expr,
                        common::ObString &prompt_value,
                        common::ObIAllocator &alloc);
  int delete_tasks(const common::ObString &task_ids);
  int execute_ai_request(const common::ObString &model_name,
                         const common::ObString &system_prompt,
                         const common::ObString &user_prompt,
                         common::ObSqlString &result);
  int write_back(const common::ObString &database_name,
                 const common::ObString &table_name,
                 const common::ObString &column_name,
                 const common::ObString &rowkey_predicate,
                 const common::ObString &value);
  int remove_task(const int64_t task_id);
  int retry_later(const int64_t task_id, const int64_t retry_count, const int64_t max_retries);
private:
  bool inited_;
  DISALLOW_COPY_AND_ASSIGN(ObAiColumnTaskWorker);
};

} // namespace observer
} // namespace oceanbase

#endif
