/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;

	double predicted_x;
	double predicted_y;
	double predicted_theta;

	for (int i = 0; i < num_particles; i++) {
		if (yaw_rate != 0) {
			predicted_x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			predicted_y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
			predicted_theta = particles[i].theta + yaw_rate*delta_t;
		} else {
			predicted_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			predicted_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			predicted_theta = particles[i].theta;
		}

		std::normal_distribution<double> dist_x(predicted_x, std_pos[0]);
		std::normal_distribution<double> dist_y(predicted_y, std_pos[1]);
		std::normal_distribution<double> dist_theta(predicted_theta, std_pos[2]);

		particles[i].x = predicted_x + dist_x(gen);
		particles[i].y = predicted_y + dist_y(gen);
		particles[i].theta = predicted_theta + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
		double dist_curr = std::numeric_limits<double>::max();
		for (int j = 0; j < predicted.size(); j++) {
			double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (d < dist_curr) {
				dist_curr = d;
				observations[i].id = predicted[j].id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		std::vector<LandmarkObs> predicted;
		//get predictions from map
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			Map::single_landmark_s l = map_landmarks.landmark_list[j];

			if (fabs(l.x_f - particles[i].x) <= sensor_range && fabs(l.y_f - particles[i].y) <= sensor_range) {
				LandmarkObs l_obs = {l.id_i, l.x_f, l.y_f};
				predicted.push_back(l_obs);
			}
		}

		//transform
		std::vector<LandmarkObs> transformed_observations(observations);
		for (int j = 0; j < transformed_observations.size(); j++) {
			transformed_observations[j].x = particles[i].x 
					+ cos(particles[i].theta) * transformed_observations[j].x 
					- sin(particles[i].theta) * transformed_observations[j].y;

			transformed_observations[j].y = particles[i].y 
					+ sin(particles[i].theta) * transformed_observations[j].x 
					+ cos(particles[i].theta) * transformed_observations[j].y;
		}

		//associate
		dataAssociation(predicted, transformed_observations);

		for (int j = 0; j < transformed_observations.size(); j++) {
			particles[i].associations.push_back(transformed_observations[j].id);
		}

		//weight
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];

		particles[i].weight = 1.0;

		for (int j = 0; j < transformed_observations.size(); j++) {

			double x, y;
			for (int k = 0; k < predicted.size(); k++) {
				if (predicted[k].id == transformed_observations[j].id) {
					x = predicted[k].x;
					y = predicted[k].y;
					break;
				}
			}

			double mu_x = transformed_observations[j].x;
			double mu_y = transformed_observations[j].y;

			double gauss_norm = 1/(2*M_PI*sig_x*sig_y);
			double exponent_1 = pow(x-mu_x,2)/(2*pow(sig_x,2));
			double exponent_2 = pow(y-mu_y,2)/(2*pow(sig_y,2));

			particles[i].weight *= gauss_norm * exp(-(exponent_1+exponent_2));
		}
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::default_random_engine gen;
	std::discrete_distribution<int> probs(weights.begin(), weights.end());
	std::vector<Particle> new_particles;
	for (int i = 0; i < num_particles; i++) {
		new_particles.push_back(particles[probs(gen)]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
