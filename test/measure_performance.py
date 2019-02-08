import subprocess
import sys
from time import sleep

run_timeout = 15 * 60

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: measure_performance.py number_of_samples /path/to/testbinary [arguments]")
		exit(1)

	number_of_samples = int(sys.argv[1])
	arguments = sys.argv[2:]

	try:
		read_data = 0
		written_data = 0
		total_time = 0.0
		times = []
		min_time = float("inf")
		max_time = float("-inf")
		additional_data = {}
		for i in range(number_of_samples):
			print("Running sample %i" % i)
			tries = 0
			while 1:
				try:
					sleep(2)
					output_raw = subprocess.check_output(arguments, timeout=run_timeout)
					break
				except subprocess.TimeoutExpired:
					tries += 1
					if tries > 10:
						print("Failed to execute kernel after 10 retries")
						exit(4)
			output = str(output_raw)
			end_of_line = output.find("csv_start\\n")
			if end_of_line == -1:
				print("Output does not have the correct format")
				exit(2)
			csv_raw = output[end_of_line:]
			rows = csv_raw.split("\\n")
			headers = rows[1].split(",")
			data = rows[2].split(",")
			if len(headers) != len(data):
				print("Output does have incorrect csv output")
				exit(3)
			for j in range(len(headers)):
				header = headers[j].strip()
				d = data[j].strip()
				if header == "time":
					t = float(d)
					total_time += t
					min_time = min(min_time, t)
					max_time = max(max_time, t)
					times.append(t)
				elif i == 0:
					if header == "read data":
						read_data = int(d)
					elif header == "written data":
						written_data = int(d)
					else:
						additional_data[header] = d

		avg_time = total_time / number_of_samples
		variance_time = 0.0
		for t in times:
			tmp = t - avg_time
			variance_time += tmp * tmp

		print("samples, read data, written data, avg time, min time, max time, variance time, %s" % ", ".join(additional_data.keys()))
		print(number_of_samples, read_data, written_data, avg_time, min_time, max_time, variance_time, sep=", ", end=", ")
		print(", ".join(additional_data.values()))

	except subprocess.CalledProcessError as e:
		print("Failed to execute command: %s" % e)