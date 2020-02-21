#ifndef BLUEFOG_COMMON_LOGGING_H
#define BLUEFOG_COMMON_LOGGING_H

#include <sstream>
#include <string>

namespace bluefog {
namespace common {

enum class LogLevel {
  TRACE, DEBUG, INFO, WARNING, ERROR, FATAL
};

#define LOG_LEVELS "TDIWEF"

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, LogLevel severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage(bool log_time);

 private:
  const char* fname_;
  int line_;
  LogLevel severity_;
};

// LogMessageFatal ensures the process will exit in failure after
// logging this message.
class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _BFG_LOG_TRACE \
  bluefog::common::LogMessage(__FILE__, __LINE__, bluefog::common::LogLevel::TRACE)
#define _BFG_LOG_DEBUG \
  bluefog::common::LogMessage(__FILE__, __LINE__, bluefog::common::LogLevel::DEBUG)
#define _BFG_LOG_INFO \
  bluefog::common::LogMessage(__FILE__, __LINE__, bluefog::common::LogLevel::INFO)
#define _BFG_LOG_WARNING \
  bluefog::common::LogMessage(__FILE__, __LINE__, bluefog::common::LogLevel::WARNING)
#define _BFG_LOG_ERROR \
  bluefog::common::LogMessage(__FILE__, __LINE__, bluefog::common::LogLevel::ERROR)
#define _BFG_LOG_FATAL \
  bluefog::common::LogMessageFatal(__FILE__, __LINE__)

#define _LOG(severity) _BFG_LOG_##severity

#define _LOG_RANK(severity, rank) _BFG_LOG_##severity << "[" << rank << "]: "

#define GET_LOG(_1, _2, NAME, ...) NAME
#define BFLOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)

LogLevel MinLogLevelFromEnv();
bool LogTimeFromEnv();

}  // namespace common
}  // namespace bluefog

#endif // BLUEFOG_COMMON_LOGGING_H
