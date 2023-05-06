import paramiko
import subprocess

result = subprocess.run(['aws', 'ec2', 'describe-instances'], capture_output=True, text=True)
output = result.stdout
output = output.split('\n')
k = False
dns_name = []
for item in output:
    if "KeyName" in item:
        if "test" in item:
            k = True
    if "PublicDnsName" in item:
        if k:
            k = False
            element = item.split(":")[1]
            if element != ' "",':
                dns_name.append(element[2:-2])

for i in range(0, len(dns_name)):
    ssh = paramiko.SSHClient()
    k = paramiko.RSAKey.from_private_key_file("~/.ssh/test.pem")
    # OR k = paramiko.DSSKey.from_private_key_file(keyfilename)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(dns_name[i])
    ssh.connect(hostname=str(dns_name[i]), username='ubuntu', pkey=k)
    # ssh.exec_command('mkdir test')
    ssh.exec_command("cd lrzipScript && /usr/bin/screen -d -m -L /home/ubuntu/anaconda3/envs/py2/bin/python history_running.py " + str(i+31))

