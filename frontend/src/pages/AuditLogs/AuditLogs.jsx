import { useState } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { FileText, Filter, Download, Eye, User, Clock } from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function AuditLogs() {
  const [filter, setFilter] = useState('all')

  const mockLogs = [
    { id: 1, action: 'User Login', user: 'john.doe@example.com', timestamp: '2024-01-15 14:30:22', ip: '192.168.1.100' },
    { id: 2, action: 'Data Export', user: 'admin@smartshoe.com', timestamp: '2024-01-15 14:25:15', ip: '10.0.0.5' },
    { id: 3, action: 'Test Result Created', user: 'system', timestamp: '2024-01-15 14:20:10', ip: 'localhost' },
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Audit Logs - Smart Shoe Monitor</title>
        <meta name="description" content="System audit logs and security monitoring" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Audit Logs</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">System activity and security monitoring</p>
          </div>
          <Button className="flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export Logs
          </Button>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card title="System Activity">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-800/50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Action</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">User</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Timestamp</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">IP Address</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {mockLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">{log.action}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">{log.user}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">{log.timestamp}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">{log.ip}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                      <Button variant="ghost" size="sm">
                        <Eye className="w-4 h-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default AuditLogs