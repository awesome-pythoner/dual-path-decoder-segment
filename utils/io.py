class Printer:
    @staticmethod
    def print_using_time(process: str, start_time: float, end_time: float) -> None:
        print(f"{process} Using Time: {int((end_time - start_time) // 3600):d}:"
              f"{int((end_time - start_time) // 60 % 60):d}:"
              f"{int((end_time - start_time) % 60):d}")
