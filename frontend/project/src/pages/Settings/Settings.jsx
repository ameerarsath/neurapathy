import { useState } from 'react'
import Card from '../../components/common/Card'
import UserSettings from './components/UserSettings'
import DeviceSettings from './components/DeviceSettings'
import NotificationSettings from './components/NotificationSettings'

function Settings() {
  const [activeTab, setActiveTab] = useState('user')
  
  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-semibold text-neutral-900">Settings</h1>
        <p className="mt-1 text-neutral-500">
          Manage your account, device, and notification preferences
        </p>
      </div>
      <div className="border-b border-neutral-200">
        <div className="flex space-x-8">
          <button
            className={`pb-4 text-sm font-medium border-b-2 ${
              activeTab === 'user'
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 hover:border-neutral-300'
            }`}
            onClick={() => setActiveTab('user')}
          >
            User Profile
          </button>
          <button
            className={`pb-4 text-sm font-medium border-b-2 ${
              activeTab === 'device'
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 hover:border-neutral-300'
            }`}
            onClick={() => setActiveTab('device')}
          >
            Device Management
          </button>
          <button
            className={`pb-4 text-sm font-medium border-b-2 ${
              activeTab === 'notification'
                ? 'border-primary-500 text-primary-600'
                : 'border-transparent text-neutral-500 hover:text-neutral-700 hover:border-neutral-300'
            }`}
            onClick={() => setActiveTab('notification')}
          >
            Notifications
          </button>
        </div>
      </div>
      <div className="mt-6">
        {activeTab === 'user' && <UserSettings />}
        {activeTab === 'device' && <DeviceSettings />}
        {activeTab === 'notification' && <NotificationSettings />}
      </div>
    </div>
  )
}

export default Settings