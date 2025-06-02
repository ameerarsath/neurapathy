import Card from '../../../components/common/Card'
import Avatar from '../../../components/common/Avatar'

function PatientInfo() {
  // Mock patient data - in a real app, this would come from an API
  const patient = {
    id: 'P123456',
    name: 'John Doe',
    dateOfBirth: 'May 15, 1958',
    age: 65,
    gender: 'Male',
    contact: {
      email: 'john.doe@example.com',
      phone: '(555) 123-4567',
      address: '123 Main St, Anytown, CA 94123'
    },
    insurance: {
      provider: 'HealthPlus',
      policyNumber: 'HP987654321',
      coverage: 'Full'
    },
    emergencyContact: {
      name: 'Jane Doe',
      relation: 'Spouse',
      phone: '(555) 987-6543'
    }
  }
  
  return (
    <Card className="h-full">
      <div className="flex flex-col items-center">
        <Avatar size="lg" name={patient.name} />
        <h2 className="mt-2 text-xl font-semibold text-neutral-900">{patient.name}</h2>
        <p className="text-neutral-500">Patient ID: {patient.id}</p>
        
        <div className="mt-4 w-full pt-4 border-t border-neutral-200">
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-neutral-500">Personal Information</h3>
              <div className="mt-2 space-y-2">
                <div className="flex justify-between">
                  <span className="text-neutral-600">Date of Birth</span>
                  <span className="text-neutral-800">{patient.dateOfBirth}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Age</span>
                  <span className="text-neutral-800">{patient.age}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Gender</span>
                  <span className="text-neutral-800">{patient.gender}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-neutral-500">Contact Information</h3>
              <div className="mt-2 space-y-2">
                <div className="flex justify-between">
                  <span className="text-neutral-600">Email</span>
                  <span className="text-neutral-800">{patient.contact.email}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Phone</span>
                  <span className="text-neutral-800">{patient.contact.phone}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-neutral-600">Address</span>
                  <span className="text-neutral-800 text-right mt-1">{patient.contact.address}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-neutral-500">Insurance</h3>
              <div className="mt-2 space-y-2">
                <div className="flex justify-between">
                  <span className="text-neutral-600">Provider</span>
                  <span className="text-neutral-800">{patient.insurance.provider}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Policy #</span>
                  <span className="text-neutral-800">{patient.insurance.policyNumber}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Coverage</span>
                  <span className="text-neutral-800">{patient.insurance.coverage}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-neutral-500">Emergency Contact</h3>
              <div className="mt-2 space-y-2">
                <div className="flex justify-between">
                  <span className="text-neutral-600">Name</span>
                  <span className="text-neutral-800">{patient.emergencyContact.name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Relation</span>
                  <span className="text-neutral-800">{patient.emergencyContact.relation}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-600">Phone</span>
                  <span className="text-neutral-800">{patient.emergencyContact.phone}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}

export default PatientInfo