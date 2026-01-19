import json
import time
from contextlib import contextmanager
from datetime import datetime


class Profiler:
    def __init__(self):
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "summary": {},
        }
        self.step_stack = []
        self.log_messages = []

    @contextmanager
    def step(self, name, **meta):
        start = time.time()
        step_data = {"name": name, "start": start, **meta}
        self.step_stack.append(step_data)
        try:
            yield
        finally:
            duration = time.time() - start
            step_data["duration"] = duration
            step_data["end"] = time.time()
            self.step_stack.pop()
            self.metrics["steps"].append(
                {
                    "name": name,
                    "duration": duration,
                    "timestamp": datetime.fromtimestamp(start).isoformat(),
                    **meta,
                }
            )

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_messages.append(log_entry)

    def record(self, key, value):
        self.metrics["summary"][key] = value

    def save(self, output_dir, name="profile"):
        import s3fs

        self.metrics["end_time"] = datetime.now().isoformat()
        total = sum(s["duration"] for s in self.metrics["steps"])
        self.metrics["summary"]["total_duration"] = total

        is_s3 = output_dir.startswith(("s3://", "s3a://"))
        base_dir = output_dir.replace("s3a://", "s3://") if is_s3 else output_dir
        profile_dir = f"{base_dir}/profile"

        json_path = f"{profile_dir}/{name}_profile.json"
        log_path = f"{profile_dir}/{name}_profile.log"

        json_content = json.dumps(self.metrics, indent=2)
        log_lines = [f"Profile Report - {self.metrics['start_time']}\n{'='*60}\n"]

        if self.log_messages:
            log_lines.append("\nLog Messages:\n")
            for msg in self.log_messages:
                log_lines.append(f"{msg}\n")
            log_lines.append(f"\n{'='*60}\n")

        log_lines.append("\nStep Durations:\n")
        for step in self.metrics["steps"]:
            meta_str = ", ".join(
                f"{k}={v}"
                for k, v in step.items()
                if k not in ["name", "duration", "timestamp"]
            )
            meta_info = f" ({meta_str})" if meta_str else ""
            log_lines.append(f"{step['name']}{meta_info}:  {step['duration']:.2f}s\n")

        log_lines.append(f"\n{'='*60}\n")
        log_lines.append(
            f"Total:  {self.metrics['summary']. get('total_duration', 0):.2f}s\n"
        )

        for key, val in self.metrics["summary"].items():
            if key != "total_duration":
                log_lines.append(f"{key}: {val}\n")

        log_content = "".join(log_lines)

        if is_s3:
            fs = s3fs.S3FileSystem()
            with fs.open(json_path, "w") as f:
                f.write(json_content)
            with fs.open(log_path, "w") as f:
                f.write(log_content)
        else:
            import os

            os.makedirs(profile_dir, exist_ok=True)
            with open(json_path, "w") as f:
                f.write(json_content)
            with open(log_path, "w") as f:
                f.write(log_content)

        print(f"\nProfile saved:  {json_path}, {log_path}")
