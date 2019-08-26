import sys

def pbar(value, endvalue, printstrings, printvalues, bar_length = 30):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{}/{} [{}] {}%".format(
        int(value),
        int(endvalue),
        arrow + spaces,
        int(round(percent * 100))))
    for i in range(len(printstrings)):
        sys.stdout.write(" - {}: {}".format(
            printstrings[i],
            printvalues[i]))
    sys.stdout.flush()
