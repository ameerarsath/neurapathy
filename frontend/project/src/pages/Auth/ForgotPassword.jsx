import { useState } from 'react'
import { Link } from 'react-router-dom'
import { supabase } from '../../services/supabase'
import Button from '../../components/common/Button'
import Input from '../../components/common/Input'
import Card from '../../components/common/Card'

function ForgotPassword() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(false)
  const [email, setEmail] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(false)

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email)
      if (error) throw error
      setSuccess(true)
    } catch (error) {
      setError(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-neutral-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-neutral-900">Reset Password</h1>
          <p className="text-neutral-600 mt-2">
            Enter your email address and we'll send you instructions to reset your password
          </p>
        </div>

        {success ? (
          <div className="text-center">
            <div className="bg-success-50 text-success-800 p-4 rounded-md mb-4">
              Password reset instructions have been sent to your email
            </div>
            <Link 
              to="/login" 
              className="text-primary-600 hover:text-primary-700 font-medium"
            >
              Return to login
            </Link>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              label="Email Address"
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />

            {error && (
              <div className="text-sm text-error-600 mt-2">{error}</div>
            )}

            <Button
              type="submit"
              className="w-full"
              isLoading={loading}
            >
              Send Reset Instructions
            </Button>

            <div className="text-center text-sm">
              <Link 
                to="/login" 
                className="text-primary-600 hover:text-primary-700 font-medium"
              >
                Back to login
              </Link>
            </div>
          </form>
        )}
      </Card>
    </div>
  )
}

export default ForgotPassword