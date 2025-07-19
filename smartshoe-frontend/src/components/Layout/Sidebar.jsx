import React from 'react'
import { NavLink } from 'react-router-dom'
import { useAuth } from '../../contexts/AuthContext'
import {
  LayoutDashboard,
  Users,
  Smartphone,
  Activity,
  Settings,
  Stethoscope,
  X,
  Heart,
  BarChart3,
  Shield,
  Brain,
  TestTube
} from 'lucide-react'
import clsx from 'clsx'

const navigation = [
  { 
    name: 'Dashboard', 
    href: '/dashboard', 
    icon: LayoutDashboard,
    roles: ['ADMIN', 'PROVIDER', 'PATIENT', 'USER']
  },
  { 
    name: 'Patient Management', 
    href: '/patients', 
    icon: Users,
    roles: ['ADMIN', 'PROVIDER']
  },
  { 
    name: 'Device Management', 
    href: '/devices', 
    icon: Smartphone,
    roles: ['ADMIN', 'PROVIDER']
  },
  { 
    name: 'Medical Readings', 
    href: '/medical-readings', 
    icon: Activity,
    roles: ['ADMIN', 'PROVIDER', 'PATIENT']
  },
  { 
    name: 'Neuropathy Testing', 
    href: '/neuropathy-testing', 
    icon: Brain,
    roles: ['ADMIN', 'PROVIDER', 'PATIENT']
  },
  { 
    name: 'ML Testing Lab', 
    href: '/ml-testing', 
    icon: TestTube,
    roles: ['ADMIN', 'PROVIDER']
  },
  { 
    name: 'Settings', 
    href: '/settings', 
    icon: Settings,
    roles: ['ADMIN', 'PROVIDER', 'PATIENT', 'USER']
  },
]

const Sidebar = ({ sidebarOpen, setSidebarOpen }) => {
  const { user, canAccess } = useAuth()

  const filteredNavigation = navigation.filter(item => 
    item.roles.some(role => canAccess(role))
  )

  return (
    <>
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 flex z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        >
          <div className="fixed inset-0 bg-neutral-600 bg-opacity-75" />
          <div className="relative flex-1 flex flex-col max-w-xs w-full bg-white">
            <div className="absolute top-0 right-0 -mr-12 pt-2">
              <button
                type="button"
                className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                onClick={() => setSidebarOpen(false)}
              >
                <X className="h-6 w-6 text-white" />
              </button>
            </div>
            <SidebarContent navigation={filteredNavigation} user={user} />
          </div>
        </div>
      )}

      {/* Desktop sidebar */}
      <div className="hidden lg:flex lg:w-64 lg:flex-col lg:fixed lg:inset-y-0">
        <div className="flex-1 flex flex-col min-h-0 bg-white border-r border-neutral-200">
          <SidebarContent navigation={filteredNavigation} user={user} />
        </div>
      </div>
    </>
  )
}

const SidebarContent = ({ navigation, user }) => {
  const getRoleColor = (role) => {
    const colors = {
      'ADMIN': 'text-primary-600 bg-primary-50',
      'PROVIDER': 'text-secondary-600 bg-secondary-50',
      'PATIENT': 'text-success bg-green-50',
      'USER': 'text-neutral-600 bg-neutral-50'
    }
    return colors[role] || colors['USER']
  }

  const getRoleIcon = (role) => {
    const icons = {
      'ADMIN': Shield,
      'PROVIDER': Stethoscope,
      'PATIENT': Heart,
      'USER': BarChart3
    }
    const Icon = icons[role] || BarChart3
    return <Icon className="h-4 w-4" />
  }

  return (
    <>
      {/* Logo and title */}
      <div className="flex items-center h-16 flex-shrink-0 px-4 bg-primary-500">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <Stethoscope className="h-8 w-8 text-white" />
          </div>
          <div className="ml-3">
            <h1 className="text-white text-lg font-semibold">Smart Shoe</h1>
            <p className="text-primary-100 text-xs">Medical Platform</p>
          </div>
        </div>
      </div>

      {/* User info */}
      <div className="px-4 py-4 border-b border-neutral-200">
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
              <span className="text-primary-600 font-medium text-sm">
                {user?.username?.charAt(0)?.toUpperCase()}
              </span>
            </div>
          </div>
          <div className="ml-3">
            <p className="text-sm font-medium text-neutral-900 capitalize">
              {user?.username}
            </p>
            <div className={clsx(
              'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium mt-1',
              getRoleColor(user?.role)
            )}>
              {getRoleIcon(user?.role)}
              <span className="ml-1">{user?.role}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
        {navigation.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                clsx(
                  isActive
                    ? 'bg-primary-50 border-primary-500 text-primary-700'
                    : 'border-transparent text-neutral-600 hover:bg-neutral-50 hover:text-neutral-900',
                  'group flex items-center px-3 py-2 text-sm font-medium border-l-4 rounded-r-md transition-colors duration-200'
                )
              }
            >
              <Icon className="mr-3 h-5 w-5 flex-shrink-0" />
              {item.name}
            </NavLink>
          )
        })}
      </nav>

      {/* System status */}
      <div className="flex-shrink-0 px-4 py-4 border-t border-neutral-200">
        <div className="flex items-center text-xs text-neutral-500">
          <div className="h-2 w-2 bg-success rounded-full mr-2"></div>
          System Online
        </div>
      </div>
    </>
  )
}

export default Sidebar