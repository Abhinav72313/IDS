import traceback
from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scapy.all import RawPcapReader, Ether, IP, TCP, UDP, ICMP


class NetworkEnvironment(gym.Env):
    def __init__(self, pcap_file: str,  max_episode_length: int = 1000):
        super().__init__()

        self.pcap_file = pcap_file
        self.max_episode_length = max_episode_length

        # Features: [pkt_size, inter_arrival, flow_duration, fwd_header_len, flow_bytes_per_sec, pkt_len_std, proto, number_of_packets, SYN, FIN, RST, PSH, ACK, URG, payload_ratio]
        self.NUM_FEATURES = 15
        self.PACKET_WINDOW = 20
        self.action_space = spaces.Discrete(2) # 0: Benign, 1: Attack
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.PACKET_WINDOW, self.NUM_FEATURES),
            dtype=np.float32
        )

        self.flows = {}
        self.MAX_FLOW_LEN = 30
        self.MAX_FLOW_SIZE = 100 # Max packets per flow to prevent memory issues

        self.pcap_reader = None
        self.reader_exhausted = False
        self.current_step = 0
        self.last_packet_ts = None
        self.packet_buffer = [] # Stores features for the sliding window

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Close previous reader if it exists
        if self.pcap_reader:
            self.pcap_reader.close()
        self.pcap_reader = RawPcapReader(self.pcap_file)
        self.reader_exhausted = False
        self.flows = {}
        self.current_step = 0
        self.last_packet_ts = None
        self.packet_buffer = []

        # Fill initial buffer with first 20 packets
        for _ in range(self.PACKET_WINDOW):
            try:
                raw_pkt, pkt_meta = next(self.pcap_reader)
                current_ts = float(pkt_meta.sec) + float(pkt_meta.usec) / 1_000_000.0
                features = self._extract_features(raw_pkt, current_ts)
                self.packet_buffer.append(features)
            except StopIteration:
                self.reader_exhausted = True
                break

        # Pad with zeros if PCAP has <20 packets
        while len(self.packet_buffer) < self.PACKET_WINDOW:
            self.packet_buffer.append(np.zeros(self.NUM_FEATURES, dtype=np.float32))

        obs = np.array(self.packet_buffer, dtype=np.float32)
        return obs, {}
        

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Single-step episode design (per DQ-IDS paper):
        1. Agent takes action based on current 20-packet window
        2. Compute reward (+1/-1 based on action vs ground truth)
        3. Advance to next 20-packet window
        4. Episode terminates after this step
        """
        # Compute reward using ground truth label (DQ-IDS Algorithm 2)
        # reward = 1.0 if action == self.label else -1.0
        reward = 0
        
        # Advance to next packet window
        terminated = False
        truncated = self.current_step >= self.max_episode_length
        
        if not self.reader_exhausted and not truncated:
            try:
                # Get next packet to shift window
                raw_pkt, pkt_meta = next(self.pcap_reader)
                current_ts = float(pkt_meta.sec) + float(pkt_meta.usec) / 1_000_000.0
                
                # Shift window: remove oldest, add newest
                if len(self.packet_buffer) == self.PACKET_WINDOW:
                    self.packet_buffer.pop(0)
                features = self._extract_features(raw_pkt, current_ts)
                self.packet_buffer.append(features)
                
            except StopIteration:
                self.reader_exhausted = True
        
        # Prepare next observation
        if len(self.packet_buffer) < self.PACKET_WINDOW:
            # Pad if near end of PCAP
            padded_buffer = self.packet_buffer + [
                np.zeros(self.NUM_FEATURES, dtype=np.float32) 
                for _ in range(self.PACKET_WINDOW - len(self.packet_buffer))
            ]
        else:
            padded_buffer = self.packet_buffer
            
        next_obs = np.array(padded_buffer, dtype=np.float32)
        self.current_step += 1
        
        # Single-step episode: always terminate after one action (DQ-IDS design)
        terminated = True
        
        return next_obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        """Render the environment."""
        # Not applicable for network packet environment
        pass

    def close(self):
        """Clean up resources."""
        if self.pcap_reader:
            self.pcap_reader.close()
        self.pcap_reader = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()

    def _extract_features(self, raw_pkt: bytes, current_ts: float) -> np.ndarray:
        """
        Extract 15 features per packet.
        Features: [pkt_size, inter_arrival, flow_duration, fwd_header_len, flow_bytes_per_sec, pkt_len_std, proto, number_of_packets, SYN, FIN, RST, PSH, ACK, URG, payload_ratio]
        """
        pkt = Ether(raw_pkt)

        # print(pkt.summary())

        # 1. Packet Size
        pkt_size = float(len(raw_pkt))

        # 2. Inter-Arrival Time (relative to *previous packet in trace*, not necessarily same flow)
        if self.last_packet_ts is None:
            inter_arrival_ts = 0.0
        else:
            inter_arrival_ts = max(current_ts - self.last_packet_ts, 0.0)
        
        self.last_packet_ts = current_ts

        # Initialize flags and header length
        tcp_flags = {'S': 0, 'F': 0, 'R': 0, 'P': 0, 'A': 0, 'U': 0}
        fwd_header_len = 0.0
        payload_ratio = 0.0

        # Extract protocol, flags, header length, payload ratio
        protocol = 0 # Default for non-IP
        if IP in pkt:
            ip_pkt = pkt[IP]
            protocol = ip_pkt.proto

            # Calculate Header Length (IP + TCP/UDP)
            ip_hdr_len = ip_pkt.ihl * 4
            l4_hdr_len = 0
            if TCP in pkt:
                tcp_pkt = pkt[TCP]
                l4_hdr_len = tcp_pkt.dataofs * 4
                # Extract TCP flags
                flags_str = str(tcp_pkt.flags)
                for flag_char in flags_str:
                    if flag_char in tcp_flags:
                        tcp_flags[flag_char] = 1.0
            elif UDP in pkt:
                l4_hdr_len = 8  # UDP header is always 8 bytes
            elif ICMP in pkt:
                 l4_hdr_len = 8  # ICMP header is 8 bytes
            fwd_header_len = float(ip_hdr_len + l4_hdr_len)

            # Calculate Payload Ratio
            total_hdr_len = ip_hdr_len + l4_hdr_len
            if pkt_size > 0:
                payload_ratio = float((pkt_size - total_hdr_len) / pkt_size)
            else:
                payload_ratio = 0.0

        # Get flow features
        flow_id = self._get_flow_id(pkt)
        if flow_id is None:
            # Non-IP packet or error getting flow ID
            flow_features = {
                'flow_duration': 0.0,
                'flow_bytes_per_sec': 0.0,
                'pkt_len_std': 0.0,
                'pkt_size_avg': 0.0,
                'packet_count': 0.0,
                'inter_arrival_flow': 0.0 # Inter-arrival within flow context if needed
            }
        else:
            self._update_flow(flow_id, pkt, current_ts)
            flow_features = self._compute_flow_features(flow_id, current_ts)

        # Features: [
        # 0. pkt_size,
        # 1. inter_arrival,
        # 2. flow_duration,
        # 3. fwd_header_len,
        # 4. flow_bytes_per_sec,
        # 5. pkt_len_std,
        # 6. proto,
        # 7. number_of_packets (packet_count),
        # 8. SYN,
        # 9. FIN,
        # 10. RST,
        # 11. PSH,
        # 12. ACK,
        # 13. URG,
        # 14. payload_ratio]

        features = np.array([
            pkt_size,
            inter_arrival_ts,
            flow_features['flow_duration'],
            fwd_header_len,
            flow_features['flow_bytes_per_sec'],
            flow_features['pkt_len_std'],
            float(protocol), # Keep as raw protocol number
            flow_features['packet_count'], # Number of packets in the flow so far
            tcp_flags['S'],
            tcp_flags['F'],
            tcp_flags['R'],
            tcp_flags['P'],
            tcp_flags['A'],
            tcp_flags['U'],
            payload_ratio
        ], dtype=np.float32)

        return features

    def _compute_flow_features(self, flow_id: Tuple, current_ts: float):
        if flow_id not in self.flows:
            return {
                'flow_duration': 0.0,
                'flow_bytes_per_sec': 0.0,
                'pkt_len_std': 0.0,
                'pkt_size_avg': 0.0,
                'packet_count': 0.0,
                'inter_arrival_flow': 0.0
            }

        flow = self.flows[flow_id]
        duration = flow['end_time'] - flow['start_time']
        if duration <= 0:
            duration = 1e-6 # Avoid division by zero

        flow_bytes_per_sec = flow['byte_count'] / duration

        if len(flow['pkt_sizes']) > 1:
            pkt_len_std = float(np.std(flow['pkt_sizes']))
            pkt_size_avg = float(np.mean(flow['pkt_sizes']))
        else:
            pkt_len_std = 0.0
            pkt_size_avg = float(flow['pkt_sizes'][0]) if flow['pkt_sizes'] else 0.0

        packet_count = float(flow['packet_count'])

        # Note: This inter_arrival is relative to the *last packet in this specific flow*
        inter_arrival_flow = current_ts - flow['last_pkt_time']

        return {
            'flow_duration': float(duration),
            'flow_bytes_per_sec': flow_bytes_per_sec,
            'pkt_len_std': pkt_len_std,
            'pkt_size_avg': pkt_size_avg,
            'packet_count': packet_count,
            'inter_arrival_flow': inter_arrival_flow,
        }

    def _update_flow(self, flow_id: Tuple, pkt, current_ts: float):
        if flow_id not in self.flows:
            self.flows[flow_id] = {
                'start_time': current_ts,
                'end_time': current_ts,
                'packet_count': 0,
                'byte_count': 0,
                'pkt_sizes': [],
                'last_pkt_time': current_ts,
            }

        flow = self.flows[flow_id]
        pkt_size = len(pkt)

        flow['end_time'] = current_ts
        flow['packet_count'] += 1
        flow['byte_count'] += pkt_size
        flow['pkt_sizes'].append(pkt_size) # Fixed typo: 'pkt_sized' -> 'pkt_sizes'
        flow['last_pkt_time'] = current_ts

        # Optional: limit memory per flow
        if len(flow['pkt_sizes']) > self.MAX_FLOW_SIZE:
            flow['pkt_sizes'] = flow['pkt_sizes'][-self.MAX_FLOW_SIZE:]

    def _get_flow_id(self, pkt):
        if IP not in pkt:
            return None

        ip = pkt[IP]
        proto = ip.proto
        src_ip, dest_ip = ip.src, ip.dst
        sport = dport = 0

        if proto == 6 and TCP in pkt:
            tcp = pkt[TCP]
            sport, dport = tcp.sport, tcp.dport
        elif proto == 17 and UDP in pkt:
            udp = pkt[UDP]
            sport, dport = udp.sport, udp.dport
        elif proto == 1 and ICMP in pkt:
             # For ICMP, use type and code as 'ports' or just 0/0, keeping IP order
             # Using 0/0 for simplicity, but type/code could be used
             sport, dport = 0, 0

        # Normalize direction: always put smaller IP first, then smaller port
        if (src_ip, sport) > (dest_ip, dport):
            src_ip, dest_ip = dest_ip, src_ip
            sport, dport = dport, sport

        return (src_ip, dest_ip, sport, dport, proto)
