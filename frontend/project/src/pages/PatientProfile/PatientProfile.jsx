import Card from '../../components/common/Card'
import Button from '../../components/common/Button'
import PatientInfo from './components/PatientInfo'
import MedicalHistory from './components/MedicalHistory'

function PatientProfile() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-neutral-900">Patient Profile</h1>
          <p className="mt-1 text-neutral-500">View and manage patient information</p>
        </div>
        <div className="mt-4 md:mt-0 flex space-x-2">
          <Button 
            variant="outline" 
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
              </svg>
            }
          >
            Edit Profile
          </Button>
          <Button
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            }
          >
            Download Report
          </Button>
        </div>
      </div>
      
      {/* Patient information cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <PatientInfo />
        <div className="lg:col-span-2">
          <MedicalHistory />
        </div>
      </div>
    </div>
  )
}

export default PatientProfile