import { useState } from 'react'
import Card from '../../../components/common/Card'
import Button from '../../../components/common/Button'

function NotificationSettings() {
  const [emailSettings, setEmailSettings] = useState({
    alerts: true,
    weeklyReports: true,
    patientUpdates: true,
    systemUpdates: false,
    marketingEmails: false
  })
  
  const [appSettings, setAppSettings] = useState({
    criticalAlerts: true,
    dataCollection: true,
    batteryAlerts: true,
    patientMessages: true,
    maintenanceReminders: true
  })
  
  const [smsSettings, setSmsSettings] = useState({
    criticalAlerts: true,
    deviceIssues: false,
    appointmentReminders: true,
    updates: false
  })
  
  const toggleEmailSetting = (setting) => {
    setEmailSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }))
  }
  
  const toggleAppSetting = (setting) => {
    setAppSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }))
  }
  
  const toggleSmsSetting = (setting) => {
    setSmsSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }))
  }
  
  return (
    <div className="space-y-6">
      <Card title="Email Notifications">
        <div className="space-y-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Patient Alerts</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Receive emails for critical patient condition alerts
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                emailSettings.alerts ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="email-alerts" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    emailSettings.alerts ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="email-alerts" 
                  name="email-alerts" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={emailSettings.alerts}
                  onChange={() => toggleEmailSetting('alerts')}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Weekly Reports</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Receive weekly summary reports of patient progress
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                emailSettings.weeklyReports ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="email-reports" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    emailSettings.weeklyReports ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="email-reports" 
                  name="email-reports" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={emailSettings.weeklyReports}
                  onChange={() => toggleEmailSetting('weeklyReports')}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Patient Updates</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Notifications when patient data changes significantly
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                emailSettings.patientUpdates ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="email-updates" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    emailSettings.patientUpdates ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="email-updates" 
                  name="email-updates" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={emailSettings.patientUpdates}
                  onChange={() => toggleEmailSetting('patientUpdates')}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">System Updates</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Notifications about software and firmware updates
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                emailSettings.systemUpdates ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="email-system" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    emailSettings.systemUpdates ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="email-system" 
                  name="email-system" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={emailSettings.systemUpdates}
                  onChange={() => toggleEmailSetting('systemUpdates')}
                />
              </div>
            </div>
          </div>
        </div>
      </Card>
      
      <Card title="App Notifications">
        <div className="space-y-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Critical Alerts</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Push notifications for urgent patient conditions
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                appSettings.criticalAlerts ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="app-critical" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    appSettings.criticalAlerts ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="app-critical" 
                  name="app-critical" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={appSettings.criticalAlerts}
                  onChange={() => toggleAppSetting('criticalAlerts')}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Data Collection Status</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Notifications about data collection status
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                appSettings.dataCollection ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="app-data" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    appSettings.dataCollection ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="app-data" 
                  name="app-data" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={appSettings.dataCollection}
                  onChange={() => toggleAppSetting('dataCollection')}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Battery Alerts</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Notifications when device batteries are low
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                appSettings.batteryAlerts ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="app-battery" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    appSettings.batteryAlerts ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="app-battery" 
                  name="app-battery" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={appSettings.batteryAlerts}
                  onChange={() => toggleAppSetting('batteryAlerts')}
                />
              </div>
            </div>
          </div>
        </div>
      </Card>
      
      <Card title="SMS Notifications">
        <div className="space-y-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Critical Alerts</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Text messages for urgent patient conditions
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                smsSettings.criticalAlerts ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="sms-critical" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    smsSettings.criticalAlerts ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="sms-critical" 
                  name="sms-critical" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={smsSettings.criticalAlerts}
                  onChange={() => toggleSmsSetting('criticalAlerts')}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Device Issues</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Text messages for device connection problems
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                smsSettings.deviceIssues ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="sms-device" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    smsSettings.deviceIssues ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="sms-device" 
                  name="sms-device" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={smsSettings.deviceIssues}
                  onChange={() => toggleSmsSetting('deviceIssues')}
                />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col md:flex-row md:items-center md:justify-between p-3 border border-neutral-200 rounded-md">
            <div>
              <h3 className="font-medium text-neutral-900">Appointment Reminders</h3>
              <p className="text-sm text-neutral-600 mt-1">
                Text message reminders for upcoming appointments
              </p>
            </div>
            <div className="mt-2 md:mt-0">
              <div className={`relative inline-block w-12 h-6 transition duration-200 ease-in-out rounded-full ${
                smsSettings.appointmentReminders ? 'bg-primary-500' : 'bg-neutral-300'
              }`}>
                <label 
                  htmlFor="sms-appointments" 
                  className={`absolute left-0 w-6 h-6 transition duration-100 ease-in-out transform bg-white rounded-full cursor-pointer ${
                    smsSettings.appointmentReminders ? 'translate-x-6' : ''
                  }`}
                ></label>
                <input 
                  type="checkbox" 
                  id="sms-appointments" 
                  name="sms-appointments" 
                  className="w-full h-full opacity-0 cursor-pointer"
                  checked={smsSettings.appointmentReminders}
                  onChange={() => toggleSmsSetting('appointmentReminders')}
                />
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 flex justify-end">
          <Button>Save Notification Preferences</Button>
        </div>
      </Card>
    </div>
  )
}

export default NotificationSettings