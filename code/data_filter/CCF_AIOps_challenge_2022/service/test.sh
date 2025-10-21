#！/bin/bash
# nohup bash test.sh >> log_drain3_test.log 2>&1 &
# nohup bash test.sh >> log_generator.log 2>&1 &
# nohup bash test.sh >> api_generator.log 2>&1 &
# nohup bash test.sh >> metric_generator.log 2>&1 &

# python log_drain3_test.py
# python api_generator.py
python metric_generator.py