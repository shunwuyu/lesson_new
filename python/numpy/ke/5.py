import numpy as np
grade = np.dtype([('name', 'S32'), ('age', 'i'), 
                  ('chinese', 'f'), 
                  ('math', 'f'), 
                  ('english', 'f')])
grades = np.array([("WangFei", 15, 86, 90, 85), ("ChenChen", 16, 87, 91, 75),
                   ("ZhangYao", 15, 86, 90, 85)], grade)
print(grades)