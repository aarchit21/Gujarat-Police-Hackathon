STEP 1 : UNDERSATND THE CHALLENGE



Background

At present, 26 different Government Departments are operating independent CCTV systems across the State. These systems include both analog and IP-based cameras deployed at geographically dispersed locations ranging from border districts to areas such as Valsad, Dahod, Somnath, Jamnagar, and Dwarka.

Each department currently operates a standalone camera ecosystem. Some departments are using cloud-based storage solutions, while others rely on local storage infrastructure. The video retention period also varies significantly, with some systems storing footage for 7 days and others for 15 days or more.

The usage and deployment pattern of cameras differ across departments based on their operational requirements and functional objectives. For example:

Cameras under the Home Department are primarily deployed in public domains for traffic monitoring, law \& order management, and crime detection.

Food \& Civil Supplies Department cameras are mainly installed at godowns, PDS shops, and related facilities.

RTO cameras are deployed at offices, testing tracks, checkpoints, and other operational locations.

The Government intends to integrate these government cameras primarily deployed in the public domain into a unified video management and analytics ecosystem. Also the proposed solution should support viewing capabilities for public-facing CCTV cameras installed by societies, malls, commercial establishments and other private entities, wherever feasible and permitted.

Moreover, various Government departments already maintain critical databases such as VAHAN, SARTHI, eGujCop (Gujarat Police's CCTNS platform), AFIS and NAFIS containing records of arrested persons, stolen vehicles, wanted criminals, missing persons, unidentified dead bodies and fingerprint data. The proposed CCTV Integration System should be integrated with these databases to enable automated real-time alerts and proactive monitoring capabilities for law enforcement agencies.

Therefore the system should be designed with a scalable, modular, and future-ready architecture to support seamless integration while ensuring secure feed exchange, standardised integration mechanisms, scalability, and compatibility.

Core Goal

Propose a secure, scalable, interoperable, technically feasible, and cost-effective approach that uses existing infrastructure to the maximum practical extent.

Key Challenges

01

Heterogeneous Infrastructure

Different vendors, VMS platforms, AMC periods, storage architectures, camera types, formats, and feed-sharing protocols.

02

Geographical Dispersion

Camera sites are distributed across the State, with distances extending to approximately 1,000 kilometers.

03

Unified Analytics

The solution should support analytics and event handling across onboarded cameras through a unified framework.

04

Scalability

New cameras, departments, systems, and future analytics should be onboarded without major redesign.







STEP 2: EXPLORE INTEGRATION MODELS



Important: Model 1 should be treated as the common CCTV registry and GIS foundation that may support Models 2, 3, and 4.



MODEL 1: Centralised CCTV Registry \& GIS Mapping Model



Introduction

Across the State, multiple departments such as Municipal Corporations, Transport Departments, Police, and various institutions have deployed CCTV cameras independently. However, there is currently no centralised mechanism to systematically identify, map, and manage these assets.



Under this model, a centralised CCTV registry and GIS-based mapping platform is proposed to be developed for onboarding and maintaining camera-related metadata such as location, department, camera type, ownership, connectivity status, storage details, and other infrastructure information. This model does not involve centralised live video streaming or recording.



The objective of this model is to create a unified inventory and visibility layer for planning future integration, identifying monitoring gaps, infrastructure assessment, and supporting decision-making. As the foundational model, it must be combined with one or more of the other proposed models to enable CCTV feed integration, unified viewing, and the generation of video analytics.



Key Functional Features

Bulk import, manual entry, and API-based camera onboarding.

Interactive GIS map with department, camera type, status, and coverage layers.

Camera health and maintenance-status monitoring.

Gap-analysis reports for uncovered zones and ageing infrastructure.

Role-based search, filtering, export, and metadata audit trails.

Suggested Technology Stack

GIS: Leaflet / OpenLayers / PostGIS

Backend: Node.js or Python (Django/FastAPI)

Database: PostgreSQL + PostGIS

Frontend: React.js

Authentication: Department-wise role-based access control

Expected Deliverables

Working registry portal with GIS map view.

Bulk and manual camera-onboarding demonstration.

Sample onboarded camera-metadata dataset.

Registry API documentation.

Sample gap-analysis report.

Diagram

Department CCTV Assets

Cameras · location · ownership

→

Onboarding \& Validation

Bulk upload · manual entry · APIs

→

Central Registry

Standardised CCTV metadata

→

PostgreSQL + PostGIS

GIS and asset data store

→

GIS Dashboard \& APIs

Map · search · reports · integration



MODEL 2: Unified Viewing \& Metadata Analytics Model



Introduction

Multiple departments currently operate independent CCTV systems through their own Video Management Systems (VMS). Central command centres are required to access these feeds through multiple separate viewer systems, which increases operational complexity and reduces monitoring efficiency.



Under this model, a unified viewing platform is proposed through which CCTV feeds from different departmental systems can be accessed through a single interface without disturbing existing infrastructure. Existing departmental VMS/storage systems shall continue to operate independently.



In this model, the proposed unified viewing platform connects directly to each departmental CCTV or VMS system through RTSP, ONVIF, vendor SDKs, or available APIs. The platform integrates the accessible video streams into a single interface without introducing an intermediate middleware or federation layer.



This model primarily focuses on centralised viewing, operational accessibility, and generation of selective metadata/analytics from accessible video streams. Depending upon technical feasibility, features such as ANPR-based metadata generation, event tagging, camera-wise indexing, and searchable vehicle movement records may also be enabled without centralised storage of all video feeds.



Key Functional Features

Feed aggregation through RTSP, ONVIF, or vendor APIs.

ANPR-based metadata generation.

Event tagging and camera-wise indexing.

Searchable vehicle-movement records.

Configurable video walls and multi-camera grid views.

Alerts for tagged events and vehicles of interest.

Suggested Technology Stack

Streaming: WebRTC / HLS relay

Integration: ONVIF / RTSP libraries / vendor SDKs

AI/ML: ANPR using open-source or custom models

Backend: Node.js or Python microservices

Messaging/Search: Kafka, Elasticsearch, PostgreSQL

Expected Deliverables

Unified viewer connected to sample feeds from at least two different systems.

ANPR demonstration on live or recorded feeds.

Searchable metadata dashboard.

Architecture note showing that existing departmental systems remain unaffected.

Diagram

Departmental VMS Platforms

Existing systems remain independent

→

RTSP / ONVIF / Vendor APIs

Secure feed access layer

→

Unified Stream Gateway

Relay · transcode · session control

→

Analytics \& Metadata

ANPR · tagging · indexing · alerts

→

Unified Control-Room View

Multi-camera viewing and search



MODEL 3: VMS Federation \& Middleware Integration Model



Introduction

Different departments across the State use heterogeneous CCTV systems supplied by multiple vendors, resulting in fragmented monitoring infrastructure and interoperability challenges.



Under this model, a middleware/federation layer is proposed to integrate multiple departmental VMS platforms through APIs, SDKs, metadata exchange, event sharing mechanisms, or standard protocols. The objective is to enable interoperability, cross-platform communication, centralized event correlation, and unified operational workflows without replacing existing departmental systems.



This model allows departments to retain their own infrastructure and operational control while enabling higher-level integration, metadata sharing, federated analytics, and coordinated monitoring capabilities at the central level.



Unlike Model 2, the proposed solution does not connect directly to each departmental CCTV or VMS system. Instead, the middleware or federation layer acts as the common integration platform, communicates with multiple existing VMS platforms, and exposes a single unified interface for downstream applications, dashboards, and AI services.



Key Functional Features

Adapter/plugin architecture for multiple VMS vendors.

Metadata exchange bus for camera and event information.

Cross-system event-correlation engine.

Unified workflow and alert dashboard.

Extensible connector framework for onboarding future vendors.

Suggested Technology Stack

Middleware: Node.js / Java (Spring Boot)

Messaging: Kafka / RabbitMQ

API Gateway: Kong / NGINX

Database: PostgreSQL and Redis

Frontend: React.js

Expected Deliverables

Working middleware demo federating at least two different systems.

Unified event-correlation dashboard.

Adapter/plugin architecture documentation.

Sample federated analytics report.

Diagram

Multiple Vendor VMS

Department-controlled source systems

→

Adapters / Connectors

API · SDK · ONVIF · protocol translation

→

Federation Middleware

Authentication · routing · orchestration

→

Event \& Metadata Bus

Kafka / RabbitMQ · correlation

→

Unified Dashboard

Cross-system monitoring and workflows



MODEL 4: Central VMS Model



Introduction

Under this model, a single consolidated Central VMS (Video Management System) is proposed to be developed for integration of CCTV cameras across various departments onto one unified platform. The Central VMS shall enable centralised monitoring, management, recording, storage, playback, and advanced analytics through a common system interface.



This model aims to create a fully integrated statewide platform capable of supporting centralised analytics, AI-based processing, vehicle tracking, event detection, and cross-departmental monitoring through a unified operational framework.



Implementation of this model would require robust infrastructure, including scalable storage, high-bandwidth connectivity, centralised compute resources, redundancy mechanisms, cybersecurity controls, and scalable architecture capable of supporting large-scale video ingestion and real-time processing.



Key Functional Features

Centralised feed ingestion.

Tiered hot, warm, and cold storage.

ANPR, face recognition, crowd/vehicle counting, and anomaly detection.

Statewide vehicle tracking and route reconstruction.

Integration readiness for VAHAN, SARTHI, eGujCop, AFIS, and NAFIS.

Redundancy, disaster recovery, encryption, network segmentation, and RBAC.

Suggested Technology Stack

VMS: Custom or extended open-source platform

Storage: S3-compatible distributed object storage / Ceph

Streaming \& inference: Kafka plus GPU-based analytics

Database: PostgreSQL / TimescaleDB

Orchestration: Kubernetes

Networking: High-bandwidth backbone with regional edge support

Expected Deliverables

Working centralised VMS prototype using sample multi-department feeds.

ANPR and multi-location vehicle-tracking demonstration.

Scalability and load-test report for approximately 80,000 cameras.

Disaster-recovery and redundancy design.

Security architecture document.

Diagram

Statewide CCTV Sources

Government and eligible integrated feeds

→

Central Ingestion Layer

Streaming gateway · load balancing

→

Central VMS Platform

Monitoring · recording · playback

→

Storage \& AI Analytics

Distributed storage · GPU inference

→

Command Centre \& Integrations

Alerts · GIS · authorised databases



MODEL 5: Hybrid / Innovative Architecture



Introduction

The four integration models described above are indicative reference models intended to guide participating companies in understanding possible approaches for CCTV integration.



Participating companies may propose a hybrid solution that combines suitable elements from two or more models, or may submit a fully innovative and customised architecture, provided the proposed solution addresses the stated functional, interoperability, security, scalability, analytics, and implementation requirements.



STEP 3: CHOOSE YOUR CHALLENGE



Using your selected reference solution model, design and build a deployment-ready solution that continuously processes the CCTV feeds provided through the hackathon portal. Your solution should demonstrate how live video streams are integrated with a searchable database of watchlist records (such as stolen vehicles, wanted persons, missing persons, blacklisted vehicles, suspect watchlists, or other entities of interest), enabling continuous Al-powered analysis and automated alert generation whenever a match is detected. Participants are expected to design the complete integration workflow, including the database structure, matching logic, alerting mechanism, and user interface. Teams should create and use representative datasets to demonstrate their solution. During the evaluation, teams may be required to identify specified vehicles or other entities from the CCTV feeds and generate accurate, real-time alerts



* Overall Architecture
* Integration Strategy
* Al \& Video Analytics
* Cybersecurity Architecture
* Deployment Architecture
* Infrastructure Sizing
* Cost-Benefit Analysis
* Department-wise Information Requirements
* Scalability Strategy
* Future Roadmap



Architecture Principles

The proposed solution shall adopt an open, modular, scalable, secure, standards-based, and vendor-neutral architecture. The solution shall avoid vendor lock-in and enable seamless integration, interoperability, replacement, upgrade, and future expansion of cameras, Video Management Systems (VMS), analytics engines, storage platforms, AI modules, and other technology components through documented standard APIs, open protocols, SDKs, and modular adapter-based frameworks. The architecture should be technology-agnostic, support heterogeneous multi-vendor environments, and facilitate future enhancements without requiring significant redesign of the overall system.



Permitted Approach

Participants may adopt any one of the following approaches, provided the solution addresses the challenge objectives of integrating diverse CCTV systems, correlating live video feeds with watchlist databases, and generating AI-powered real-time alerts for law enforcement.

Any one of the five reference solution models provided in the official Problem Statement.

A hybrid architecture combining features from two or more reference solution models.

A fully customised architecture designed to meet the challenge requirements.



STEP 4: TECHNICAL EVALUATION / TEST CASES



Live Challenge

Integrate approximately 50 heterogeneous cameras, track a designated vehicle and deliver real-time insights, alerts and GIS visualization.

* Onboard \~50 cameras
* Integrate live feeds
* Track vehicle
* Generate alerts \& history
* Visualize on GIS
* Route, movement history \& searchable events



Test Scenario

After completing the registration, participants will be able to access the details, resources, and live camera feeds from approximately 50 geographically distributed cameras through the Resources page of the website for the technical evaluation.

The cameras are deployed across different departments and use various technologies, formats, VMS platforms, and storage mechanisms.

Teams must onboard the available cameras onto one integrated platform.

The solution must enable centralised monitoring and AI-powered video analytics.

During the evaluation, participants will be provided with a designated vehicle registration number. The solution must demonstrate its capability to identify, trace, and present the movement of the corresponding vehicle across the integrated CCTV network as it appears at different camera locations and times.

The solution should also demonstrate continuous cross-referencing of live CCTV feeds with a representative watchlist database and generate automated real-time alerts upon detecting a match. Participants may use their own representative watchlist database for this demonstration.

Expected Output

Demonstration of the solution's capability to identify and trace the designated vehicle across the integrated CCTV network using the vehicle registration number provided during the evaluation.

Complete route traversed by the designated vehicle, including timestamped and location-wise movement history.

Demonstration of a working watchlist database integrated with the solution, showcasing continuous cross-referencing between live CCTV feeds and representative watchlist records, along with automated real-time alert generation upon detecting a match. Participants may create and use their own representative watchlist database for this demonstration.

Evidence of successful CCTV integration, AI-powered video analytics, interoperability, scalability, and end-to-end system performance.



STEP 5: PREPARE TO SUBMIT



Deliverables

Submit the prototype, documentation, architecture details and any additional inputs as per hackathon guidelines.

* Solution Presentation
* High-Level Design Document
* Own-Feed Demonstration
* Government-Feed Demonstration
* Video \& Output Report
* Submission Links



Submission Requirements

Documents

1\. Solution Presentation (PPT/PDF)

Proposed solution model (Reference Model 1–5, Hybrid, or Customised Architecture) with justification.

Solution overview, objectives, and key innovations.

High-level system architecture and end-to-end workflow.

AI-powered video analytics approach, including detection, recognition, and event analytics.

Methodology for correlating live CCTV feeds with watchlist databases and generating automated real-time alerts.

Key technologies, frameworks, and tools used.

Scalability, interoperability, security, and deployment considerations.

Expected operational benefits and impact on policing and public safety.

Documents

2\. Technical Proposal — High-Level Design (HLD)

Overall solution architecture, including high-level architecture diagrams and component interactions.

Approach for integrating heterogeneous CCTV cameras, NVRs, and Video Management Systems (VMS) into a unified platform.

Architecture for ingesting, processing, and managing live video streams from geographically dispersed camera locations.

Approach for integrating live CCTV feeds with watchlist databases (e.g., stolen vehicles, wanted persons, missing persons, blacklisted vehicles, suspect watchlists) and continuously correlating video analytics results to generate real-time alerts.

AI-powered video analytics approach, including technologies such as Automatic Number Plate Recognition (ANPR), Facial Recognition Systems (FRS), object detection, person and vehicle tracking, and other intelligent analytics proposed by the team.

Alert generation and notification workflow, including prioritisation, visualisation, and user interaction.

Scalability, interoperability, security, and performance considerations for statewide deployment and future expansion to approximately 80,000 cameras.

Technical prerequisites, assumptions, and information required from participating departments to assess camera integration feasibility and ensure seamless interoperability.

Demonstrations

3\. Demonstration on Participant's Own Feed

Submit a screen-recorded demonstration (maximum 2–3 minutes) showcasing your solution operating on CCTV cameras or video footage of your choice. The demonstration should clearly illustrate:

Onboarding and processing of live or recorded CCTV feeds.

AI-powered detection and analytics (such as ANPR, Facial Recognition, or other proposed capabilities).

Correlation of detected entities with a representative watchlist database (e.g., stolen vehicles, wanted persons, missing persons, blacklisted vehicles, or suspect watchlists).

Automatic generation of real-time alerts and visualisation upon successful detection and database matching.

The demonstration must showcase a fullUy functional working solution. Mock-ups, animations, simulated interfaces, or concept videos without an operational backend will not be considered.

Demonstrations

4\. Live Demonstration on Government-Provided CCTV Feed

Onboard the Government-provided feed(s) onto the proposed platform.

Demonstrate successful onboarding and live or recorded viewing.

Demonstrate available video-analytics output on the provided feed.

Submit a screen-recorded video along with an output report showing detected vehicles or number plates with corresponding timestamps.

How to Submit

Submission Method

Unlisted YouTube link, with video visibility set to Unlisted.

Google Drive or OneDrive link with access enabled for "Anyone with the link — Viewer".

Participants may additionally provide a URL to their hosted platform along with test login credentials for the screening committee.

Participants may also provide a GitHub or GitLab repository link containing the platform source code or relevant components.





STEP 6: PLAN FOR SCALE



Scalability Requirement

Present your strategy to scale the solution securely and reliably to approximately 80,000 cameras across Gujarat.

* Hardware \& Software Requirements
* Network \& Bandwidth Planning
* Storage \& Retention Strategy
* Al Processing Capacity
* Disaster Recovery Strategy
* Statewide Rollout Plan



Participants should explain

* Central, regional, and edge-compute requirements.
* GPU or accelerator requirements for video analytics.
* Expected network bandwidth and low-bandwidth strategies.
* Hot, warm, and cold storage assumptions based on retention periods.
* Load balancing, horizontal scaling, monitoring, logging, and health checks.
* High availability, backup, disaster recovery, and cybersecurity controls.
* Estimated implementation and operational costs.



STEP 7: EVALUATION \& RECOGNITION



Get Evaluated \& Recognised

Solutions will be evaluated qualitatively across multiple dimensions. Innovative solutions may earn special recognition.

* Successful Test Case
* PPT/PDF Presentation
* Solution Architecture
* Working Demonstration
* Analytics Quality
* Scalability \& POC Readiness
* Bonus Consideration



Evaluation Framework

All eligible submissions will first be assessed against the common evaluation areas. Bonus consideration may then be awarded for meaningful capabilities demonstrated beyond the mandatory requirements.



A. Common Evaluation Areas

01

Successful Test Case

Successful onboarding and operation of the solution on the Government-provided CCTV feed, including live or recorded viewing and the required analytics output.



02

Solution Presentation

Clarity and completeness of the submitted PPT or PDF, including problem understanding, proposed model, justification, solution overview, and key features.



03

Solution Architecture

Technical soundness, feasibility, security, interoperability, and clarity of the proposed High-Level Design and architecture diagrams.



04

Working Platform and Demonstration

Maturity of the actual working platform demonstrated through the participant's own feed and the Government-provided feed.



05

Video Analytics Output

Quality and usefulness of ANPR, vehicle or person detection, intrusion detection, object detection, timestamps, and output reports.



06

Scalability and PoC Readiness

Readiness to scale toward approximately 80,000 cameras and preparedness for the on-site proof-of-concept.



07

Submission Completeness

Completeness, accessibility, and consistency of all required documents, videos, reports, links, credentials, and supporting technical information.



B. Bonus Consideration

Bonus consideration may be given for additional capabilities that are relevant, functional, and demonstrated as part of the working solution. Bonus features will not compensate for failure to meet any mandatory submission or test requirement.



* Innovative hybrid or customised architecture with clear operational value.
* Advanced cross-camera vehicle movement tracking or multi-camera correlation.
* Additional reliable analytics beyond the mandatory ANPR requirement.
* Strong edge-processing, bandwidth-optimisation, or low-connectivity operation.
* Enhanced cybersecurity, privacy protection, auditability, or role-based access controls.
* Operational dashboards, automated alerts, health monitoring, or integration-ready APIs.

