from datetime import datetime

log_buffer = []

def log(message):
    print(message)
    log_buffer.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

def get_log_buffer():
    return log_buffer


def clear_log_buffer():
    global log_buffer
    log_buffer.clear()