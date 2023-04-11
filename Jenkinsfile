pipeline {
  agent any
  stages {
    stage('Build') {
      parallel {
        stage('Build') {
          post {
            success {
              echo 'Build Success!'
            }

          }
          steps {
            echo 'Building..'
            sh 'make build'
          }
        }

        stage('Test') {
          steps {
            echo 'Testing branch'
          }
        }

      }
    }

  }
}