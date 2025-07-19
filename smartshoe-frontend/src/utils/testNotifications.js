// Test utility to add sample notifications
export const addTestNotifications = (notificationContext) => {
  const { showSuccess, showError, showWarning, showInfo, showMedicalAlert, showDeviceAlert } = notificationContext
  
  // Add various test notifications
  showSuccess('Test connection successful', { 
    title: 'Connection Test' 
  })
  
  showInfo('System maintenance scheduled', { 
    title: 'Maintenance Notice' 
  })
  
  showWarning('Device battery low', { 
    title: 'Battery Warning' 
  })
  
  showMedicalAlert('Abnormal reading detected', { 
    title: 'Medical Alert' 
  })
  
  showDeviceAlert('Device calibration required', { 
    title: 'Device Alert' 
  })
  
  showError('Connection failed', { 
    title: 'Connection Error' 
  })
}

// Function to test notification system
export const testNotificationSystem = () => {
  console.log('Testing notification system...')
  
  // Test with mock data
  const mockNotifications = [
    {
      id: 1,
      title: 'Welcome',
      message: 'Welcome to Smart Shoe Dashboard',
      type: 'success',
      timestamp: new Date().toISOString(),
      read: false
    },
    {
      id: 2,
      title: 'Device Update',
      message: 'Your device firmware has been updated',
      type: 'info',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      read: true
    },
    {
      id: 3,
      title: 'Medical Alert',
      message: 'Abnormal neuropathy test results detected',
      type: 'medical_alert',
      timestamp: new Date(Date.now() - 7200000).toISOString(),
      read: false
    }
  ]
  
  return mockNotifications
}