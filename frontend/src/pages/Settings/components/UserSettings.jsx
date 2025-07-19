import { useState } from 'react'
import Card from '../../../components/common/Card'
import Input from '../../../components/common/Input'
import Button from '../../../components/common/Button'
import Avatar from '../../../components/common/Avatar'

function UserSettings() {
  const [user, setUser] = useState({
    name: 'Dr. Alex Smith',
    email: 'alex.smith@medical.com',
    phone: '(555) 123-4567',
    specialty: 'Endocrinology',
    licenseNumber: 'MED12345',
    hospital: 'City Medical Center',
    department: 'Diabetes Care'
  })
  
  const handleChange = (e) => {
    const { name, value } = e.target
    setUser(prev => ({ ...prev, [name]: value }))
  }
  
  const handleSubmit = (e) => {
    e.preventDefault()
    // In a real app, you would submit the form data to an API
    alert('Profile updated successfully!')
  }
  
  return (
    <div className="space-y-6">
      <Card title="Profile Information">
        <div className="flex flex-col md:flex-row md:items-start">
          <div className="flex flex-col items-center mb-6 md:mb-0 md:mr-8">
            <Avatar size="lg" name={user.name} />
            <button className="mt-4 text-sm text-primary-600 hover:text-primary-700 font-medium">
              Change Photo
            </button>
          </div>
          
          <form className="flex-1" onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Input
                label="Full Name"
                id="name"
                name="name"
                value={user.name}
                onChange={handleChange}
                required
              />
              
              <Input
                label="Email Address"
                type="email"
                id="email"
                name="email"
                value={user.email}
                onChange={handleChange}
                required
              />
              
              <Input
                label="Phone Number"
                id="phone"
                name="phone"
                value={user.phone}
                onChange={handleChange}
              />
              
              <Input
                label="Specialty"
                id="specialty"
                name="specialty"
                value={user.specialty}
                onChange={handleChange}
              />
              
              <Input
                label="License Number"
                id="licenseNumber"
                name="licenseNumber"
                value={user.licenseNumber}
                onChange={handleChange}
              />
              
              <Input
                label="Hospital"
                id="hospital"
                name="hospital"
                value={user.hospital}
                onChange={handleChange}
              />
              
              <Input
                label="Department"
                id="department"
                name="department"
                value={user.department}
                onChange={handleChange}
              />
            </div>
            
            <div className="mt-6 flex justify-end">
              <Button type="submit">Save Changes</Button>
            </div>
          </form>
        </div>
      </Card>
      
      <Card title="Account Security">
        <form>
          <div className="space-y-4">
            <Input
              label="Current Password"
              type="password"
              id="currentPassword"
              name="currentPassword"
              placeholder="Enter your current password"
            />
            
            <Input
              label="New Password"
              type="password"
              id="newPassword"
              name="newPassword"
              placeholder="Enter your new password"
              helperText="Password must be at least 8 characters and include a number and symbol"
            />
            
            <Input
              label="Confirm New Password"
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              placeholder="Confirm your new password"
            />
          </div>
          
          <div className="mt-6 flex justify-end">
            <Button variant="outline" className="mr-3">Cancel</Button>
            <Button>Update Password</Button>
          </div>
        </form>
      </Card>
      
      <Card title="Two-Factor Authentication" className="border-warning-200">
        <div className="flex items-start">
          <div className="bg-warning-100 p-2 rounded-full mr-4">
            <svg className="w-6 h-6 text-warning-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          
          <div className="flex-1">
            <h3 className="text-base font-medium text-neutral-900">Protect your account</h3>
            <p className="mt-1 text-sm text-neutral-600">
              Add an extra layer of security to your account by enabling two-factor authentication.
              We'll send a verification code to your phone each time you sign in.
            </p>
            <div className="mt-4">
              <Button>Enable 2FA</Button>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}

export default UserSettings