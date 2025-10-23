from scapy.all import IP
from cicflowmeter.flow_session import generate_session_class
from decimal import Decimal
import tempfile
import os

def scapy_packets_to_flows(packets, save_csv=None):
    
    if not packets:
        # print("No packets provided")
        return []
    
    # Ensure packets have proper timestamps and fields
    processed_packets = []

    for pkt in packets:
        if IP not in pkt:
            continue  # Skip non-IP packets
            
        # Ensure required fields are set
        
        pkt.time = Decimal(str(pkt.time))
            
        # Ensure IP header length is set (required by cicflowmeter)
        if pkt[IP].ihl is None:
            pkt[IP].ihl = 5  # Standard 20-byte header
            
        processed_packets.append(pkt)
    
    if not processed_packets:
        return []
    
    
    # Create temporary CSV file if none specified
    temp_csv = save_csv
    if not temp_csv:
        temp_fd, temp_csv = tempfile.mkstemp(suffix='.csv')
        os.close(temp_fd)
    
    try:
        # Create FlowSession
        FlowSessionClass = generate_session_class(
            output_mode="flow",
            output_file=temp_csv,
            url_model=None
        )
        
        session = FlowSessionClass()
        
        # Process each packet
        success_count = 0
        for packet in processed_packets:
            try:
                session.on_packet_received(packet)
                success_count += 1
            except Exception as e:
                # print(f"Packet processing error: {e}")
                pass
        
        # print(f" Successfully processed {success_count}/{len(processed_packets)} packets")
        
        # Get flows before garbage collection
        flows = list(session.get_flows())
        
        # Trigger CSV generation
        if save_csv:
            session.garbage_collect(None)
            # print(f"💾 CSV saved to: {save_csv}")
        
        return flows
        
    except Exception as e:
        # print(f"❌ Flow extraction error: {e}")
        return []
    
    finally:
        # Cleanup temporary file if created
        if not save_csv and os.path.exists(temp_csv):
            os.unlink(temp_csv)
