
from ids_peak import ids_peak as peak


class camera:
    def __init__(self,excosure,frame):
        self.excosure=excosure
        self.frame=frame
        print("Start")
        print("press q to exite camera")



    def get_frame(self):

        self.frame = 30
        

    # initialize library
        peak.Library.Initialize()
    # Create instance of the device manager
        device_manager = peak.DeviceManager.Instance()

        try:
        # Update the device manager
            device_manager.Update()
            if device_manager.Devices().empty():
                print("No device found")
            m_device=device_manager.Devices()[0].OpenDevice(peak.DeviceAccessType_Control)
            data_streams = m_device.DataStreams()
            m_dataStream = data_streams[0].OpenDataStream()
            m_node_map_remote_device=m_device.RemoteDevice().NodeMaps()[0]
            try:
                m_node_map_remote_device.FindNode("UserSetSelector").SetCurrentEntry("Default")
                m_node_map_remote_device.FindNode("UserSetLoad").Execute()
                m_node_map_remote_device.FindNode("UserSetLoad").WaitUntilDone()
                m_node_map_remote_device.FindNode("ExposureTime").SetValue(self.excosure)
                
            except peak.Exception:
                # Userset is not available
                pass
            payload_size = m_node_map_remote_device.FindNode("PayloadSize").Value()
            num_buffers_min_required = m_dataStream.NumBuffersAnnouncedMinRequired() + 1
            for count in range(num_buffers_min_required):
                buffer = m_dataStream.AllocAndAnnounceBuffer(payload_size)
                m_dataStream.QueueBuffer(buffer)
            try:
                max_fps = m_node_map_remote_device.FindNode("AcquisitionFrameRate").Maximum()
                target_fps = min(max_fps, self.frame)
                m_node_map_remote_device.FindNode("AcquisitionFrameRate").SetValue(target_fps)
            except peak.Exception:
                # Userset is not available
                pass
        #acquisition_timer.setInterval((1 / target_fps) * 1000)
        #acquisition_timer.setSingleShot(False)
        #acquisition_timer.timeout.connect(on_acquisition_timer)

            try:
            # Lock critical features to prevent them from changing during acquisition
                m_node_map_remote_device.FindNode("TLParamsLocked").SetValue(1)

            # Start acquisition on camera
                m_dataStream.StartAcquisition()
                m_node_map_remote_device.FindNode("AcquisitionStart").Execute()
                m_node_map_remote_device.FindNode("AcquisitionStart").WaitUntilDone()
            except Exception as e:
                print("CAN NOT FOUND: critical features")

            return buffer,m_dataStream,m_device,m_node_map_remote_device,target_fps,self.excosure
            print("hi")
        except Exception as e:
            # ...
            str_error = str(e)
