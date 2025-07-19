import { useState } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { Video, Phone, MessageSquare, Calendar, User, Clock } from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function Telemedicine() {
  const [activeCall, setActiveCall] = useState(false)

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Telemedicine - Smart Shoe Monitor</title>
        <meta name="description" content="Connect with healthcare providers remotely" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Telemedicine</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">Connect with your healthcare providers remotely</p>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card title="Video Consultation">
            <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center relative">
              {!activeCall ? (
                <div className="text-center">
                  <Video className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-xl font-medium text-white mb-2">Ready to connect</h3>
                  <p className="text-gray-300 mb-6">Click "Start Call" to begin your consultation</p>
                  <Button onClick={() => setActiveCall(true)} className="bg-green-600 hover:bg-green-700">
                    <Video className="w-4 h-4 mr-2" />
                    Start Call
                  </Button>
                </div>
              ) : (
                <div className="absolute inset-4 bg-blue-600 rounded-lg flex items-center justify-center">
                  <div className="text-center text-white">
                    <User className="w-24 h-24 mx-auto mb-4 opacity-50" />
                    <p className="text-lg">Dr. Smith</p>
                    <p className="text-sm opacity-75">Connected</p>
                  </div>
                  <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-4">
                    <Button variant="ghost" className="bg-red-600 hover:bg-red-700" onClick={() => setActiveCall(false)}>
                      End Call
                    </Button>
                    <Button variant="ghost" className="bg-gray-600 hover:bg-gray-700">
                      <MessageSquare className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <Card title="Upcoming Sessions">
            <div className="space-y-3">
              <div className="p-3 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="flex items-center gap-3 mb-2">
                  <User className="w-4 h-4 text-blue-600" />
                  <span className="font-medium text-gray-900 dark:text-white">Dr. Smith</span>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <div className="flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    Today, 2:00 PM
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    30 minutes
                  </div>
                </div>
                <Button size="sm" className="w-full mt-3">Join Now</Button>
              </div>
            </div>
          </Card>

          <Card title="Quick Actions">
            <div className="space-y-3">
              <Button variant="outline" className="w-full justify-start">
                <Video className="w-4 h-4 mr-2" />
                Schedule Consultation
              </Button>
              <Button variant="outline" className="w-full justify-start">
                <MessageSquare className="w-4 h-4 mr-2" />
                Send Message
              </Button>
              <Button variant="outline" className="w-full justify-start">
                <Phone className="w-4 h-4 mr-2" />
                Request Callback
              </Button>
            </div>
          </Card>

          <Card title="Connection Status">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Internet</span>
                <span className="text-sm font-medium text-green-600">Excellent</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Camera</span>
                <span className="text-sm font-medium text-green-600">Ready</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Microphone</span>
                <span className="text-sm font-medium text-green-600">Ready</span>
              </div>
            </div>
          </Card>
        </div>
      </motion.div>
    </div>
  )
}

export default Telemedicine