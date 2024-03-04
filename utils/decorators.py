from time import time


def timer(process="Task"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time()
            output = func(*args, **kwargs)
            end_time = time()
            print(f"{process} Using Time: {int((end_time - start_time) // 3600):d}:"
                  f"{int((end_time - start_time) // 60 % 60):d}:"
                  f"{int((end_time - start_time) % 60):d}")
            return output

        return wrapper

    return decorator


def repeat(times=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return [func(*args, **kwargs) for _ in range(times)]

        return wrapper

    return decorator
