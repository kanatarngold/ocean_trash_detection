import subprocess

def run_git(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return f"CMD: {command}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n{'-'*20}\n"
    except Exception as e:
        return f"CMD: {command}\nERROR: {e}\n{'-'*20}\n"

output = ""
output += run_git("git status")
output += run_git("git log -n 1")
output += run_git("git remote -v")

with open("git_debug_output.txt", "w") as f:
    f.write(output)
