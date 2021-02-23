# Zeigt den Inhalt der .pkl Pickle Dateien an
from batchgenerators.utilities.file_and_folder_operations import *
# Retina 2D
print(30*"-"+"Augen-Adern")
a = load_pickle('plans-Augen-203.pkl')
print(a['plans_per_stage'])

# Larven
print(30*"-"+"Larven Drittel Split")
a = load_pickle('plans-Larven-201.pkl')
print(a['plans_per_stage'])

# 3D Heart
print(30*"-"+"Heart")
a = load_pickle('plans-Heart-103.pkl')
print(a['plans_per_stage'])

# Pascal VOC 2012
print(30*"-"+"Pascal")
a = load_pickle('plans-Pascal-204.pkl')
print(a['plans_per_stage'])