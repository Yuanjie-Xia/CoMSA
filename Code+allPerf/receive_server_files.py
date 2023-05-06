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

print(dns_name)
for i in range(len(dns_name)):
    command = "scp -i ~/.ssh/test.pem ubuntu@" + dns_name[i] + ":~/commit_running_time* /Users/yuanjiexia/lrzipScript/perf_data"
    print(i)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).communicate()[0]