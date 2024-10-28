y_length = 10
y_start = 1
y_reverse = True

def y_print(y_length, y_start, y_reverse):
  i = y_start
  while i <= y_length:
    print(i)
    i += 1
  if y_reverse:
    while i >= y_start:
      print(i)
      i -= 1

y_print(y_length, y_start, y_reverse)
