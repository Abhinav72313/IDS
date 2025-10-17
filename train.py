from NetworkEnvironment import NetworkEnvironment

benign_env = NetworkEnvironment(pcap_file='BenignTraffic3.pcap')


obs,info = benign_env.reset()

while True:
    action = benign_env.action_space.sample()
    next_state, reward, terminated, truncated, _ = benign_env.step(action)

    print(next_state,end='\n\n')

    if truncated :
        break
