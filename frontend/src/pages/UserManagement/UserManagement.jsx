import { useState } from 'react'
import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  Users, 
  Plus, 
  Search, 
  Filter,
  Edit,
  Trash2,
  UserCheck,
  UserX,
  Shield
} from 'lucide-react'

import { api } from '@services/api'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import Input from '@components/common/Input'
import LoadingSpinner from '@components/common/LoadingSpinner'

function UserManagement() {
  const [searchTerm, setSearchTerm] = useState('')
  const [roleFilter, setRoleFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')

  const { data: users, isLoading } = useQuery(
    ['users', searchTerm, roleFilter, statusFilter],
    () => api.admin.getUsers({ search: searchTerm, role: roleFilter, status: statusFilter }),
    {
      staleTime: 2 * 60 * 1000,
    }
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading users..." />
      </div>
    )
  }

  const getRoleBadgeColor = (role) => {
    switch (role?.toLowerCase()) {
      case 'admin': return 'bg-red-100 text-red-800'
      case 'provider': return 'bg-blue-100 text-blue-800'
      case 'caregiver': return 'bg-green-100 text-green-800'
      case 'patient': return 'bg-gray-100 text-gray-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getStatusBadgeColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'active': return 'bg-green-100 text-green-800'
      case 'inactive': return 'bg-red-100 text-red-800'
      case 'pending': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const mockUsers = [
    { id: 1, name: 'John Doe', email: 'john@example.com', role: 'PATIENT', status: 'ACTIVE', lastLogin: '2024-01-15' },
    { id: 2, name: 'Dr. Smith', email: 'smith@hospital.com', role: 'PROVIDER', status: 'ACTIVE', lastLogin: '2024-01-15' },
    { id: 3, name: 'Jane Wilson', email: 'jane@example.com', role: 'CAREGIVER', status: 'ACTIVE', lastLogin: '2024-01-14' },
    { id: 4, name: 'Admin User', email: 'admin@smartshoe.com', role: 'ADMIN', status: 'ACTIVE', lastLogin: '2024-01-15' },
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>User Management - Smart Shoe Monitor</title>
        <meta name="description" content="Manage user accounts and permissions" />
      </Helmet>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            User Management
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Manage user accounts, roles, and permissions
          </p>
        </div>
        
        <Button className="mt-4 lg:mt-0 flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Add User
        </Button>
      </motion.div>

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row gap-4"
      >
        <div className="flex-1">
          <Input
            type="text"
            placeholder="Search users..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
            icon={Search}
          />
        </div>
        
        <select
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="all">All Roles</option>
          <option value="ADMIN">Admin</option>
          <option value="PROVIDER">Provider</option>
          <option value="CAREGIVER">Caregiver</option>
          <option value="PATIENT">Patient</option>
        </select>
        
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="all">All Status</option>
          <option value="ACTIVE">Active</option>
          <option value="INACTIVE">Inactive</option>
          <option value="PENDING">Pending</option>
        </select>
      </motion.div>

      {/* User Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-4 gap-4"
      >
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <Users className="w-5 h-5 text-blue-600" />
            <div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {mockUsers.length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Total Users
              </div>
            </div>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <UserCheck className="w-5 h-5 text-green-600" />
            <div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {mockUsers.filter(u => u.status === 'ACTIVE').length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Active Users
              </div>
            </div>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <Shield className="w-5 h-5 text-purple-600" />
            <div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {mockUsers.filter(u => u.role === 'PROVIDER' || u.role === 'ADMIN').length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Staff Members
              </div>
            </div>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <UserX className="w-5 h-5 text-red-600" />
            <div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {mockUsers.filter(u => u.status === 'INACTIVE').length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Inactive Users
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Users Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card title="Users" className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-800/50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    User
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Role
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Last Login
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {mockUsers.map((user, index) => (
                  <motion.tr
                    key={user.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 * index }}
                    className="hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div>
                        <div className="text-sm font-medium text-gray-900 dark:text-white">
                          {user.name}
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          {user.email}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getRoleBadgeColor(user.role)}`}>
                        {user.role}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusBadgeColor(user.status)}`}>
                        {user.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {new Date(user.lastLogin).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex justify-end gap-2">
                        <Button variant="ghost" size="sm">
                          <Edit className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="sm" className="text-red-600 hover:text-red-700">
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default UserManagement