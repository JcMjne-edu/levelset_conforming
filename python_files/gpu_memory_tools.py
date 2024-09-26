import subprocess

DEFAULT_ATTRIBUTES_MEMORY = (
    'memory.total',
    'memory.free',
    'memory.used',
)
def print_gpu_memory(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES_MEMORY, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    lines=lines[0].split(', ')
    print(f'{float(lines[1])/1000} GB free / {float(lines[0])/1000} GB total')
