import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../../contexts/AuthContext'
import { useNotifications } from '../../contexts/NotificationContext'
import { 
  Menu, 
  Bell, 
  Settings, 
  LogOut, 
  User,
  ChevronDown,
  Activity
} from 'lucide-react'
import clsx from 'clsx'

const Header = ({ setSidebarOpen }) => {
  const { user, logout } = useAuth()
  const { notifications } = useNotifications()
  const navigate = useNavigate()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [notificationDropdownOpen, setNotificationDropdownOpen] = useState(false)
  
  const unreadNotifications = notifications.filter(n => !n.read).length

  return (
    <div className="bg-white shadow-sm border-b border-neutral-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Left side */}
          <div className="flex items-center">
            {/* Mobile menu button */}
            <button
              type="button"
              className="lg:hidden -ml-0.5 -mt-0.5 h-12 w-12 inline-flex items-center justify-center rounded-md text-neutral-500 hover:text-neutral-900 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu className="h-6 w-6" />
            </button>

            {/* Page title area */}
            <div className="hidden lg:flex lg:items-center lg:space-x-4">
              <div>
                <h1 className="text-xl font-semibold text-neutral-900">
                  Smart Shoe Dashboard
                </h1>
                <p className="text-sm text-neutral-500">
                  Diabetic Neuropathy Monitoring Platform
                </p>
              </div>
            </div>
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-4">
            {/* System status indicator */}
            <div className="hidden sm:flex items-center space-x-2 text-sm">
              <div className="flex items-center text-success">
                <Activity className="h-4 w-4 mr-1" />
                <span>Live</span>
              </div>
            </div>

            {/* Notifications */}
            <div className="relative">
              <button 
                className="p-2 text-neutral-400 hover:text-neutral-500 relative"
                onClick={() => setNotificationDropdownOpen(!notificationDropdownOpen)}
              >
                <Bell className="h-5 w-5" />
                {unreadNotifications > 0 && (
                  <span className="absolute top-1 right-1 block h-2 w-2 rounded-full bg-error"></span>
                )}
              </button>
              
              {/* Notification Dropdown */}
              {notificationDropdownOpen && (
                <div className="origin-top-right absolute right-0 mt-2 w-80 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-50">
                  <div className="py-1">
                    <div className="px-4 py-2 text-sm text-neutral-500 border-b">
                      Notifications ({unreadNotifications} unread)
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      {notifications.length === 0 ? (
                        <div className="px-4 py-3 text-sm text-neutral-500 text-center">
                          No notifications
                        </div>
                      ) : (
                        notifications.slice(0, 10).map((notification) => (
                          <div
                            key={notification.id}
                            className={clsx(
                              "px-4 py-3 text-sm border-b border-neutral-100 hover:bg-neutral-50",
                              !notification.read && "bg-blue-50"
                            )}
                          >
                            <div className="flex items-center justify-between">
                              <p className="font-medium text-neutral-900">
                                {notification.title || notification.message}
                              </p>
                              <span className="text-xs text-neutral-500">
                                {new Date(notification.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                            {notification.message && notification.title && (
                              <p className="text-neutral-600 mt-1">
                                {notification.message}
                              </p>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* User dropdown */}
            <div className="relative">
              <button
                className="flex items-center space-x-2 text-sm rounded-md p-2 hover:bg-neutral-50"
                onClick={() => setDropdownOpen(!dropdownOpen)}
              >
                <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                  <span className="text-primary-600 font-medium text-sm">
                    {user?.username?.charAt(0)?.toUpperCase()}
                  </span>
                </div>
                <div className="hidden sm:block text-left">
                  <p className="font-medium text-neutral-900 capitalize">
                    {user?.username}
                  </p>
                  <p className="text-xs text-neutral-500">{user?.role}</p>
                </div>
                <ChevronDown className="h-4 w-4 text-neutral-400" />
              </button>

              {/* Dropdown menu */}
              {dropdownOpen && (
                <div className="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-50">
                  <div className="py-1">
                    <button 
                      className="flex items-center px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100 w-full text-left"
                      onClick={() => {
                        navigate('/patients/' + (user?.id || '1'))
                        setDropdownOpen(false)
                      }}
                    >
                      <User className="mr-3 h-4 w-4" />
                      Profile
                    </button>
                    <button 
                      className="flex items-center px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100 w-full text-left"
                      onClick={() => {
                        navigate('/settings')
                        setDropdownOpen(false)
                      }}
                    >
                      <Settings className="mr-3 h-4 w-4" />
                      Settings
                    </button>
                    <div className="border-t border-neutral-100"></div>
                    <button
                      onClick={logout}
                      className="flex items-center px-4 py-2 text-sm text-error hover:bg-neutral-100 w-full text-left"
                    >
                      <LogOut className="mr-3 h-4 w-4" />
                      Sign out
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Close dropdowns when clicking outside */}
      {dropdownOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setDropdownOpen(false)}
        />
      )}
      {notificationDropdownOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setNotificationDropdownOpen(false)}
        />
      )}
    </div>
  )
}

export default Header