import subprocess 

cmd = 'python StoreData.py'

# The number of runs you want.
runs = 5


# Runs the StoreData.py 'runs' amount of times. 
# Settings runs to 10 will result in 10 numpy's 
# assuming no errors occur.
while (runs > 0):
    p = subprocess.Popen(cmd, shell=True)

    out, err = p.communicate()
    print(err)
    print(out)

    runs = runs - 1