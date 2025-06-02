import { useState } from 'react'
import Card from '../../../components/common/Card'
import Button from '../../../components/common/Button'

function DeviceSettings() {
  const [devices, setDevices] = useState([
    { 
      id: 1, 
      name: 'Smart Insole (Right)', 
      serialNumber: 'SI-R-12345', 
      status: 'Connected', 
      batteryLevel: 87,
      lastSync: '2023-12-28T14:30:00',
      firmwareVersion: '2.3.1',
      isActive: true
    },
    { 
      id: 2, 
      name: 'Smart Insole (Left)', 
      serialNumber: 'SI-L-12346', 
      status: 'Connected', 
      batteryLevel: 92,
      lastSync: '2023-12-28T14:30:00',
      firmwareVersion: '2.3.1',
      isActive: true
    },
    { 
      id: 3, 
      name: 'Pressure Monitor', 
      serialNumber: 'PM-78901', 
      status: 'Disconnected', 
      batteryLevel: 45,
      lastSync: '2023-12-25T09:15:00',
      firmwareVersion: '1.8.5',
      isActive: false
    }
  ])
  
  const [showAddDevice, setShowAddDevice] = useState(false)
  
  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  
  const getBatteryColor = (level) => {
    if (level >= 60) return 'text-success-600'
    if (level >= 30) return 'text-warning-600'
    return 'text-error-600'
  }
  
  const toggleDeviceStatus = (id) => {
    setDevices(devices.map(device => 
      device.id === id ? { ...device, isActive: !device.isActive } : device
    ))
  }
  
  return (
    <div className="space-y-6">
      <Card 
        title="Paired Devices"
        footer={
          <div className="flex justify-end">
            <Button onClick={() => setShowAddDevice(!showAddDevice)}>
              Add New Device
            </Button>
          </div>
        }
      >
        <div className="space-y-4">
          {devices.map(device => (
            <div 
              key={device.id} 
              className={`p-4 border rounded-lg transition ${
                device.isActive 
                  ? 'border-primary-200 bg-primary-50' 
                  : 'border-neutral-200 bg-neutral-50'
              }`}
            >
              <div className="flex flex-col md:flex-row md:items-center md:justify-between">
                <div className="mb-3 md:mb-0">
                  <div className="flex items-center">
                    <h3 className="text-lg font-medium text-neutral-900">{device.name}</h3>
                    <span className={`ml-2 px-2 py-0.5 text-xs rounded-full ${
                      device.status === 'Connected' 
                        ? 'bg-success-100 text-success-800' 
                        : 'bg-neutral-100 text-neutral-800'
                    }`}>
                      {device.status}
                    </span>
                  </div>
                  <p className="text-sm text-neutral-500 mt-1">Serial: {device.serialNumber}</p>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="flex items-center">
                    <div className={`mr-1 ${getBatteryColor(device.batteryLevel)}`}>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 5h12a2 2 0 012 2v10a2 2 0 01-2 2H3a2 2 0 01-2-2V7a2 2 0 012-2z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11h5.5a1.5 1.5 0 011.5 1.5v0a1.5 1.5 0 01-1.5 1.5H15v-3z" />
                      </svg>
                    </div>
                    <span className={`text-sm ${getBatteryColor(device.batteryLevel)}`}>
                      {device.batteryLevel}%
                    </span>
                  </div>
                  
                  <Button 
                    size="sm" 
                    variant={device.isActive ? 'primary' : 'outline'}
                    onClick={() => toggleDeviceStatus(device.id)}
                  >
                    {device.isActive ? 'Active' : 'Inactive'}
                  </Button>
                </div>
              </div>
              
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-neutral-500">Last Synced:</span>
                  <span className="ml-2 text-neutral-700">{formatDate(device.lastSync)}</span>
                </div>
                <div>
                  <span className="text-neutral-500">Firmware:</span>
                  <span className="ml-2 text-neutral-700">{device.firmwareVersion}</span>
                </div>
                <div>
                  <Button variant="ghost" size="sm" className="text-xs">Update Firmware</Button>
                </div>
              </div>
            </div>
          ))}
          
          {showAddDevice && (
            <div className="p-4 border border-dashed border-neutral-300 rounded-lg bg-neutral-50 flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 bg-neutral-200 rounded-full flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-neutral-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-neutral-900">Add a New Device</h3>
                <p className="text-sm text-neutral-600 mt-1 mb-4">
                  Make sure your device is in pairing mode and within range
                </p>
                <Button>Scan for Devices</Button>
              </div>
            </div>
          )}
        </div>
      </Card>
      
      <Card title="Data Collection Settings">
        <div className="space-y-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Pressure Readings</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Collect pressure data from insoles
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className="relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full bg-primary-500">
                <label htmlFor="pressure" className="absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer translate-x-6"></label>
                <input 
                  type="checkbox" 
                  id="pressure" 
                  name="pressure" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  defaultChecked={true}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Temperature Monitoring</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Track temperature variations across foot surfaces
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className="relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full bg-primary-500">
                <label htmlFor="temperature" className="absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer translate-x-6"></label>
                <input 
                  type="checkbox" 
                  id="temperature" 
                  name="temperature" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  defaultChecked={true}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Motion Analysis</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Track gait and movement patterns throughout the day
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className="relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full bg-neutral-300">
                <label htmlFor="motion" className="absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer"></label>
                <input 
                  type="checkbox" 
                  id="motion" 
                  name="motion" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  defaultChecked={false}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Continuous Monitoring</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Collect data 24/7 even when not actively exercising
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className="relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full bg-primary-500">
                <label htmlFor="continuous" className="absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer translate-x-6"></label>
                <input 
                  type="checkbox" 
                  id="continuous" 
                  name="continuous" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  defaultChecked={true}
                />
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 flex justify-end">
          <Button>Save Settings</Button>
        </div>
      </Card>
    </div>
  )
}

export default DeviceSettings