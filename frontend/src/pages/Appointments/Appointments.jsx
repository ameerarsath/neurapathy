import { useState } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { Calendar, Plus, Clock, User, Video } from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function Appointments() {
  const mockAppointments = [
    { id: 1, date: '2024-01-20', time: '10:00 AM', doctor: 'Dr. Smith', type: 'Consultation', status: 'scheduled' },
    { id: 2, date: '2024-01-25', time: '2:00 PM', doctor: 'Dr. Johnson', type: 'Follow-up', status: 'scheduled' },
    { id: 3, date: '2024-02-01', time: '11:00 AM', doctor: 'Dr. Brown', type: 'Telemedicine', status: 'scheduled' },
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Appointments - Smart Shoe Monitor</title>
        <meta name="description" content="Manage your medical appointments and consultations" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Appointments</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">Manage your upcoming medical appointments</p>
          </div>
          <Button className="flex items-center gap-2">
            <Plus className="w-4 h-4" />
            Schedule Appointment
          </Button>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card title="Upcoming Appointments">
            <div className="space-y-4">
              {mockAppointments.map((apt, index) => (
                <div key={apt.id} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                      {apt.type === 'Telemedicine' ? <Video className="w-6 h-6 text-blue-600" /> : <User className="w-6 h-6 text-blue-600" />}
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900 dark:text-white">{apt.doctor}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{apt.type}</p>
                      <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                        <span className="flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          {apt.date}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {apt.time}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">Reschedule</Button>
                    {apt.type === 'Telemedicine' && <Button size="sm">Join Call</Button>}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        <div>
          <Card title="Quick Actions">
            <div className="space-y-3">
              <Button variant="outline" className="w-full justify-start">
                <Calendar className="w-4 h-4 mr-2" />
                Schedule New Appointment
              </Button>
              <Button variant="outline" className="w-full justify-start">
                <Video className="w-4 h-4 mr-2" />
                Request Telemedicine
              </Button>
              <Button variant="outline" className="w-full justify-start">
                <Clock className="w-4 h-4 mr-2" />
                View Calendar
              </Button>
            </div>
          </Card>
        </div>
      </motion.div>
    </div>
  )
}

export default Appointments