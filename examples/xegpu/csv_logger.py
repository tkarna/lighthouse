import logging
import csv
import os


class CSVLogger:
    def __init__(self, filename=None, append=False):
        self.filename = filename
        self.header_written = False
        self.fieldnames = None
        self.logger = logging.getLogger("csv_logger")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        if self.filename is not None and not append:
            assert not os.path.exists(self.filename), (
                f"CSV file '{self.filename}' already exists"
            )
        self.loaded = self._load() if append else None
        if self.loaded:
            print(f"Loaded {len(self.loaded) if self.loaded else 0} existing entries")

    def log(self, data: dict):
        if self.fieldnames is None:
            self.fieldnames = list(data.keys())
        write_header = not os.path.exists(self.filename) or not self.header_written
        if write_header:
            self.logger.info(",".join(self.fieldnames))
        self.logger.info(",".join(str(data[k]) for k in self.fieldnames))
        if self.filename is None:
            return
        with open(self.filename, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(data)

    def _load(self):
        # load existing CSV file as one string per row
        with open(self.filename, mode="r") as csvfile:
            # strip lines and create dict
            data = [line.strip() for line in csvfile.readlines()]
        fieldnames = data[0].split(",")
        data = {tuple(line.split(",")): 1 for line in data[1:]}
        data = {",".join(d[:-2]): 1 for d in data}
        if self.fieldnames is None:
            self.fieldnames = list(fieldnames)
        return data

    def contains(self, entry: dict):
        if not self.loaded:
            raise RuntimeError("No data loaded. Call _load() first.")
        # check if data (except last two elements) is in existing CSV file
        key = ",".join(str(entry[k]) for k in self.fieldnames)
        return key in self.loaded
