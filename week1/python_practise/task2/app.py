alter = 17.9
threshold = 18
volljährig = "Volljährig"
minderjährig = "Minderjährig"

def decide_if_adult(alter, threshold, volljährig, minderjährig):
  if alter >= 18:
    print(volljährig)
  else:
    print(minderjährig)

decide_if_adult(alter, threshold, volljährig, minderjährig)