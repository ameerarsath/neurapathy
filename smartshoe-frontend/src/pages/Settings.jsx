import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import TwoFactorSetup from '../components/auth/TwoFactorSetup'
import { 
  Settings as SettingsIcon, 
  User, 
  Bell, 
  Shield, 
  Database,
  Monitor,
  Save
} from 'lucide-react'

const Settings = () => {
  const { user } = useAuth()
  const [activeTab, setActiveTab] = useState('profile')

  const tabs = [
    { id: 'profile', name: 'Profile', icon: User },
    { id: 'notifications', name: 'Notifications', icon: Bell },
    { id: 'security', name: 'Security', icon: Shield },
    { id: 'data', name: 'Data & Privacy', icon: Database },
    { id: 'system', name: 'System', icon: Monitor },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="medical-card">
        <div className="flex items-center space-x-3">
          <SettingsIcon className="h-6 w-6 text-primary-600" />
          <div>
            <h1 className="text-2xl font-bold text-neutral-900">Settings</h1>
            <p className="text-neutral-600 mt-1">
              Manage your account and system preferences
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <div className="medical-card">
            <nav className="space-y-1">
              {tabs.map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center space-x-3 px-3 py-2 text-left text-sm font-medium rounded-md transition-colors ${
                      activeTab === tab.id
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{tab.name}</span>
                  </button>
                )
              })}
            </nav>
          </div>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-3">
          {activeTab === 'profile' && <ProfileSettings user={user} />}
          {activeTab === 'notifications' && <NotificationSettings />}
          {activeTab === 'security' && <SecuritySettings />}
          {activeTab === 'data' && <DataPrivacySettings />}
          {activeTab === 'system' && <SystemSettings />}
        </div>
      </div>
    </div>
  )
}

const ProfileSettings = ({ user }) => {
  return (
    <div className="medical-card">
      <h3 className="text-lg font-medium text-neutral-900 mb-6">Profile Settings</h3>
      <form className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="form-group">
            <label className="form-label">Username</label>
            <input
              type="text"
              className="form-input"
              value={user?.username || ''}
              disabled
            />
          </div>
          <div className="form-group">
            <label className="form-label">Role</label>
            <input
              type="text"
              className="form-input"
              value={user?.role || ''}
              disabled
            />
          </div>
        </div>
        
        <div className="form-group">
          <label className="form-label">Email</label>
          <input
            type="email"
            className="form-input"
            placeholder="Enter your email"
          />
        </div>
        
        <div className="form-group">
          <label className="form-label">Full Name</label>
          <input
            type="text"
            className="form-input"
            placeholder="Enter your full name"
          />
        </div>
        
        <button type="submit" className="flex items-center px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors">
          <Save className="h-4 w-4 mr-2" />
          Save Changes
        </button>
      </form>
    </div>
  )
}

const NotificationSettings = () => {
  return (
    <div className="medical-card">
      <h3 className="text-lg font-medium text-neutral-900 mb-6">Notification Preferences</h3>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-neutral-900">Critical Alerts</h4>
            <p className="text-sm text-neutral-500">Immediate notifications for critical medical readings</p>
          </div>
          <input type="checkbox" className="rounded border-neutral-300" defaultChecked />
        </div>
        
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-neutral-900">Device Notifications</h4>
            <p className="text-sm text-neutral-500">Alerts for device status and connectivity</p>
          </div>
          <input type="checkbox" className="rounded border-neutral-300" defaultChecked />
        </div>
        
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-neutral-900">Daily Reports</h4>
            <p className="text-sm text-neutral-500">Summary of daily activity and readings</p>
          </div>
          <input type="checkbox" className="rounded border-neutral-300" />
        </div>
      </div>
    </div>
  )
}

const SecuritySettings = () => {
  return (
    <div className="space-y-6">
      {/* Password Change */}
      <div className="medical-card">
        <h3 className="text-lg font-medium text-neutral-900 mb-6">Change Password</h3>
        <div className="space-y-6">
          <div className="form-group">
            <label className="form-label">Current Password</label>
            <input type="password" className="form-input" />
          </div>
          
          <div className="form-group">
            <label className="form-label">New Password</label>
            <input type="password" className="form-input" />
          </div>
          
          <div className="form-group">
            <label className="form-label">Confirm New Password</label>
            <input type="password" className="form-input" />
          </div>
          
          <button className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors duration-200">
            <Save className="h-4 w-4 mr-2" />
            Change Password
          </button>
        </div>
      </div>
      
      {/* Two-Factor Authentication */}
      <TwoFactorSetup />
    </div>
  )
}

const DataPrivacySettings = () => {
  return (
    <div className="medical-card">
      <h3 className="text-lg font-medium text-neutral-900 mb-6">Data & Privacy</h3>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-neutral-900">Data Sharing</h4>
            <p className="text-sm text-neutral-500">Allow anonymized data for research purposes</p>
          </div>
          <input type="checkbox" className="rounded border-neutral-300" />
        </div>
        
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-neutral-900">Analytics</h4>
            <p className="text-sm text-neutral-500">Help improve the platform with usage analytics</p>
          </div>
          <input type="checkbox" className="rounded border-neutral-300" defaultChecked />
        </div>
        
        <div className="pt-4 border-t border-neutral-200">
          <button className="text-error hover:text-red-700 text-sm font-medium">
            Export My Data
          </button>
        </div>
      </div>
    </div>
  )
}

const SystemSettings = () => {
  return (
    <div className="medical-card">
      <h3 className="text-lg font-medium text-neutral-900 mb-6">System Settings</h3>
      <div className="space-y-6">
        <div className="form-group">
          <label className="form-label">Language</label>
          <select className="form-input">
            <option>English</option>
            <option>Spanish</option>
            <option>French</option>
          </select>
        </div>
        
        <div className="form-group">
          <label className="form-label">Timezone</label>
          <select className="form-input">
            <option>America/New_York</option>
            <option>America/Los_Angeles</option>
            <option>Europe/London</option>
          </select>
        </div>
        
        <div className="form-group">
          <label className="form-label">Date Format</label>
          <select className="form-input">
            <option>MM/DD/YYYY</option>
            <option>DD/MM/YYYY</option>
            <option>YYYY-MM-DD</option>
          </select>
        </div>
      </div>
    </div>
  )
}

export default Settings