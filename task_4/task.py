import cv2
import numpy as np
import pandas as pd
import scipy
import glob

from sklearn.neighbors import KNeighborsClassifier

def floodfill(mat, y, x, c1, c2):
  upper_bound = (x,y)
  lower_bound = (x,y)
  stack = [(y,x)]
  while len(stack) > 0:
    ny, nx = stack.pop()
    if ny >= len(mat) or ny < 0 or nx >= len(mat[ny]) or nx < 0:
      continue
    if mat[ny][nx] != c1:
      continue
    upper_bound = (min(upper_bound[0], nx), min(upper_bound[1], ny))
    lower_bound = (max(lower_bound[0], nx), max(lower_bound[1], ny))
    mat[ny][nx] = c2
    stack.append((ny + 1, nx))
    stack.append((ny - 1, nx))
    stack.append((ny, nx + 1))
    stack.append((ny, nx - 2))
  return (upper_bound, lower_bound)
  
def check_border(mat, y, x):
  height, width = mat.shape
  if mat[y][x] == 0:
    return False
  if y == 0 or x == 0 or y == height - 1 or x == width - 1:
    return True
  if 0 in [mat[y+1][x], mat[y-1][x], mat[y][x+1], mat[y][x-1]]:
    return True
  return False

def generate_chain_code(mat, y, x):
  cluster_id = mat[y][x]
  dx = [0, 1, 1, 1, 0, -1, -1, -1]
  dy = [-1, -1, 0, 1, 1, 1, 0, -1]

  result = []
  pos = (x,y)
  while mat[pos[1]][pos[0]] == cluster_id:
    mat[pos[1]][pos[0]] *= -1
    old_pos = pos
    for d, temp in enumerate(zip(dx, dy)):
      ix, iy = temp
      nx, ny = pos[0] + ix, pos[1] + iy
      if nx >= 0 and ny >= 0 and ny < len(mat) and nx < len(mat[ny]) and mat[ny][nx] == cluster_id and check_border(mat, ny, nx):
        result.append(d)
        pos = (nx, ny)
        break
    if pos == old_pos:
      break
  return result

def stretch_chain_code(chain_code, len_dest):
  ret = []
  if len(chain_code) < len_dest:
    scale =  len_dest / len(chain_code)
    for i in range(len_dest):
      ret += [chain_code[min(round(i/scale),len(chain_code)-1)]]
  else:
    scale = len(chain_code) / len_dest
    for i in range(len_dest):
      i_from = round(i*scale)
      i_to = min(round(i_from + scale), len(chain_code))
      # ret += [max(set(chain_code[i_from:i_to]), key=chain_code[i_from:i_to].count)]
      ret += [int(round(np.average(chain_code[i_from:i_to])))]
  return ret

def generate_all_chain_code(mat):
  height, width = mat.shape

  boundaries = {}
  cluster = 0
  for y in range(height):
    for x in range(width):
      if mat[y][x] == -1:
        cluster += 1
        boundaries[cluster] = floodfill(mat, y, x, -1, cluster)
        
  chain_codes = {}
  for y in range(height):
    for x in range(width):
      if mat[y][x] > 0 and abs(mat[y][x]) not in chain_codes and check_border(mat, y, x):
        cluster_id = mat[y][x]
        chain_code = generate_chain_code(mat, y, x)
        chain_codes[cluster_id] = chain_code

  indices = list(filter(lambda x: len(chain_codes[x]) > 0, range(1, cluster + 1)))
  return [boundaries[i] for i in indices], [chain_codes[i] for i in indices]

def get_all_chain_codes_in_image(image):
  threshold = 110
  image_mat = np.vectorize(lambda x: -1 if x < threshold else 0)(image)
  return generate_all_chain_code(image_mat)

data_x = []
data_y = []
for file in glob.glob('training/*'):
    image = cv2.imread(file)
    image_color = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    x, chain_codes = get_all_chain_codes_in_image(image_grayscale)
    chain_code = max(map(lambda x: (len(x), x), chain_codes))[1]
    chain_code = stretch_chain_code(chain_code, 180)
    
    label = file[9]
    data_x.append(chain_code)
    data_y.append(int(label))
    
print("Template chain codes")
for number, chain_codes in zip(data_y, data_x):
    print("%d: %s" % (number, "".join(map(lambda x: str(x), chain_codes))))

knn = KNeighborsClassifier(n_neighbors=1, metric= 'manhattan')
knn.fit(data_x, data_y)

cap = cv2.VideoCapture(0)
while True:
  ret, image = cap.read()

  height, width, _ = image.shape
  image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  boundaries, chain_codes = get_all_chain_codes_in_image(image_grayscale)
  chain_codes = list(map(lambda y: stretch_chain_code(y, 180), chain_codes))

  predicted = knn.predict(chain_codes)
  for boundary, chain_code, result in zip(boundaries, chain_codes, predicted):
    upper_bound, lower_bound = boundary
    area = abs(lower_bound[1] - upper_bound[1]) * abs(lower_bound[0] - upper_bound[0])
    if area < height * width * 0.001:
      continue
    
    # print("%s predicted as %d" % ("".join(map(lambda x: str(x), chain_code)), result))
    cv2.rectangle(image, upper_bound, lower_bound, (255,0,0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(result), (upper_bound[0], upper_bound[1] + 10), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

  cv2.imshow('raw image', image)
  if cv2.waitKey(1) & 0xFF == ord('q'):
  	break

cap.release()
cv2.destroyAllWindows()