// Test script to verify device dropdown functionality
const BASE_URL = 'http://localhost:8080';

// Test function to verify device API
async function testDeviceAPI() {
    console.log('🔍 Testing Device API...');
    
    const auth = btoa('admin:admin123');
    
    try {
        const response = await fetch(`${BASE_URL}/api/devices`, {
            headers: {
                'Authorization': `Basic ${auth}`,
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('✅ Device API Response:', data);
        
        if (data.success && data.devices) {
            console.log(`📦 Found ${data.devices.length} devices:`);
            data.devices.forEach((device, index) => {
                console.log(`   ${index + 1}. ${device.model} (${device.serialNumber}) - Status: ${device.status}`);
            });
            
            // Test dropdown filtering logic
            const activeDevices = data.devices.filter(d => d.status === 'ACTIVE' || d.status === 'LOW_BATTERY');
            console.log(`🔌 Active/Low Battery devices for dropdown: ${activeDevices.length}`);
            
            if (activeDevices.length > 0) {
                console.log('✅ Device dropdown should show options');
                activeDevices.forEach((device, index) => {
                    console.log(`   Option ${index + 1}: ${device.model} - ${device.serialNumber}`);
                });
            } else {
                console.log('❌ No active devices found for dropdown');
            }
        } else {
            console.log('❌ Device API returned no devices');
        }
    } catch (error) {
        console.error('❌ Error testing device API:', error);
    }
}

// Test function to verify patient API
async function testPatientAPI() {
    console.log('\n🔍 Testing Patient API...');
    
    const auth = btoa('admin:admin123');
    
    try {
        const response = await fetch(`${BASE_URL}/api/patients`, {
            headers: {
                'Authorization': `Basic ${auth}`,
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('✅ Patient API Response:', data);
        
        if (data.success && data.patients) {
            console.log(`👥 Found ${data.patients.length} patients:`);
            data.patients.forEach((patient, index) => {
                console.log(`   ${index + 1}. ${patient.firstName} ${patient.lastName} - ${patient.email}`);
            });
            console.log('✅ Patient dropdown should show options');
        } else {
            console.log('❌ Patient API returned no patients');
        }
    } catch (error) {
        console.error('❌ Error testing patient API:', error);
    }
}

// Test function to verify neuropathy test creation
async function testNeuropathyTestCreation() {
    console.log('\n🔍 Testing Neuropathy Test Creation...');
    
    const auth = btoa('admin:admin123');
    
    const testData = {
        patientId: 1,
        deviceId: 1,
        footSide: 'LEFT',
        isBaseline: false
    };
    
    try {
        const response = await fetch(`${BASE_URL}/api/neuropathy/test/start`, {
            method: 'POST',
            headers: {
                'Authorization': `Basic ${auth}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(testData)
        });
        
        const data = await response.json();
        console.log('✅ Neuropathy Test Creation Response:', data);
        
        if (data.success) {
            console.log(`✅ Test created successfully with ID: ${data.testId}`);
            console.log(`📊 Test details: ${data.totalStimuli} stimuli, ${data.estimatedDuration} minutes`);
        } else {
            console.log('❌ Test creation failed:', data.message);
        }
    } catch (error) {
        console.error('❌ Error testing neuropathy test creation:', error);
    }
}

// Run all tests
async function runAllTests() {
    console.log('🚀 Starting Device Dropdown Integration Tests...\n');
    
    await testDeviceAPI();
    await testPatientAPI();
    await testNeuropathyTestCreation();
    
    console.log('\n✅ All tests completed!');
}

// Run the tests
runAllTests();