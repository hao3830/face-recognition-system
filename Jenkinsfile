/* groovylint-disable-next-line CompileStatic */
pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                sh 'make build'
            }
            post {
                success {
                    echo 'Build Success!'
                }
            }
        }
    }
}
