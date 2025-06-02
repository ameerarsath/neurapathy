import Card from '../../../components/common/Card'
import Button from '../../../components/common/Button'

function MedicalHistory() {
  // Mock medical history data - in a real app, this would come from an API
  const conditions = [
    { name: 'Type 2 Diabetes', diagnosedDate: '2010-05-12', status: 'Active', notes: 'Currently managed with medication and lifestyle changes.' },
    { name: 'Hypertension', diagnosedDate: '2012-11-03', status: 'Active', notes: 'Blood pressure typically ranges from 135/85 to 145/90.' },
    { name: 'Peripheral Neuropathy', diagnosedDate: '2015-07-22', status: 'Active', notes: 'Affects both feet, worse in the right foot.' },
  ]
  
  const medications = [
    { name: 'Metformin', dosage: '1000mg', frequency: 'Twice daily', startDate: '2010-05-15', endDate: null },
    { name: 'Lisinopril', dosage: '10mg', frequency: 'Once daily', startDate: '2012-11-10', endDate: null },
    { name: 'Gabapentin', dosage: '300mg', frequency: 'Three times daily', startDate: '2015-08-05', endDate: null },
  ]
  
  const labTests = [
    { name: 'HbA1c', date: '2023-11-15', value: '7.2%', range: '<6.5%', status: 'High' },
    { name: 'Blood Pressure', date: '2023-11-15', value: '138/88', range: '<130/80', status: 'High' },
    { name: 'Cholesterol', date: '2023-11-15', value: '195 mg/dL', range: '<200 mg/dL', status: 'Normal' },
    { name: 'Creatinine', date: '2023-11-15', value: '1.0 mg/dL', range: '0.7-1.3 mg/dL', status: 'Normal' },
  ]
  
  const appointments = [
    { date: '2023-12-15', provider: 'Dr. Smith', type: 'Regular Check-up', notes: 'Follow-up on medication adjustment' },
    { date: '2023-10-05', provider: 'Dr. Jones', type: 'Podiatrist', notes: 'Foot examination, minor callus treatment' },
    { date: '2023-08-20', provider: 'Dr. Smith', type: 'Regular Check-up', notes: 'HbA1c review, adjusted metformin dose' },
  ]
  
  return (
    <div className="space-y-6">
      {/* Medical conditions */}
      <Card
        title="Medical Conditions"
        footer={
          <div className="flex justify-end">
            <Button variant="ghost" size="sm">View All</Button>
          </div>
        }
      >
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-neutral-200">
            <thead>
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Condition</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Diagnosed</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Status</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200">
              {conditions.map((condition, index) => (
                <tr key={index}>
                  <td className="px-3 py-2 text-sm font-medium text-neutral-900">{condition.name}</td>
                  <td className="px-3 py-2 text-sm text-neutral-600">{condition.diagnosedDate}</td>
                  <td className="px-3 py-2 text-sm">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-success-100 text-success-800">
                      {condition.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-sm text-neutral-600">{condition.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      
      {/* Medications */}
      <Card
        title="Current Medications"
        footer={
          <div className="flex justify-end">
            <Button variant="ghost" size="sm">View All</Button>
          </div>
        }
      >
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-neutral-200">
            <thead>
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Medication</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Dosage</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Frequency</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Started</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200">
              {medications.map((medication, index) => (
                <tr key={index}>
                  <td className="px-3 py-2 text-sm font-medium text-neutral-900">{medication.name}</td>
                  <td className="px-3 py-2 text-sm text-neutral-600">{medication.dosage}</td>
                  <td className="px-3 py-2 text-sm text-neutral-600">{medication.frequency}</td>
                  <td className="px-3 py-2 text-sm text-neutral-600">{medication.startDate}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      
      {/* Recent lab results */}
      <Card
        title="Recent Lab Results"
        footer={
          <div className="flex justify-end">
            <Button variant="ghost" size="sm">View All</Button>
          </div>
        }
      >
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-neutral-200">
            <thead>
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Test</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Date</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Result</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Reference Range</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200">
              {labTests.map((test, index) => (
                <tr key={index}>
                  <td className="px-3 py-2 text-sm font-medium text-neutral-900">{test.name}</td>
                  <td className="px-3 py-2 text-sm text-neutral-600">{test.date}</td>
                  <td className="px-3 py-2 text-sm text-neutral-600">{test.value}</td>
                  <td className="px-3 py-2 text-sm text-neutral-600">{test.range}</td>
                  <td className="px-3 py-2 text-sm">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      test.status === 'Normal' 
                        ? 'bg-success-100 text-success-800' 
                        : 'bg-warning-100 text-warning-800'
                    }`}>
                      {test.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      
      {/* Past appointments */}
      <Card
        title="Recent Appointments"
        footer={
          <div className="flex justify-end">
            <Button variant="ghost" size="sm">View All</Button>
          </div>
        }
      >
        <div className="space-y-4">
          {appointments.map((appointment, index) => (
            <div key={index} className="p-3 border border-neutral-200 rounded-md">
              <div className="flex flex-col sm:flex-row sm:justify-between mb-2">
                <div className="font-medium text-neutral-900">{appointment.date}</div>
                <div className="text-neutral-600">{appointment.provider} • {appointment.type}</div>
              </div>
              <p className="text-sm text-neutral-600">{appointment.notes}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

export default MedicalHistory