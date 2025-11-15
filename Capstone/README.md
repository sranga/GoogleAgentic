# **Project Overview – Vaccine-Access Agent (V-Access)**

The **Vaccine-Access Agent (V-Access)** is a multi-agent healthcare-support system designed to make vaccine information, scheduling, and follow-up accessible to underserved communities. Leveraging Google’s Agent Development Kit (ADK), the system orchestrates several specialized agents, tools, and workflows to educate users, locate clinics, complete scheduling steps, and ensure that follow-up doses are not missed.

The system aims to reduce health disparities by providing a reliable, conversational, multilingual assistant that can operate across chat, SMS, and low-bandwidth environments.


## **Problem Statement**

Accessing vaccines is surprisingly hard for many individuals. Barriers include:

* a lack of clear and trustworthy information

* confusion around eligibility, doses, or side-effects

* difficulty locating nearby clinics

* challenges scheduling appointments on fragmented portals

* missed second doses due to busy schedules or forgetfulness

These barriers disproportionately affect underserved communities, deepening healthcare inequities.

Even when information exists, it is:

* poorly structured

* hard to navigate

* not localized

* not available in multiple languages

Users may feel overwhelmed, uncertain, or distrustful — leading to reduced vaccine uptake and preventable health risks.


## **Solution Statement**

**V-Access** uses a coordinated multi-agent system to streamline the entire vaccine journey. The system can:

* educate users using plain language in their preferred language

* answer questions about vaccine types, risks, and side-effects

* find nearby clinics using geolocation tools

* schedule appointments (integrating with OpenAPI booking systems where available)

* send follow-up reminders using long-running operations

* check in post-vaccination and escalate if symptoms require attention

* store user-specific progress with session memory

* generate anonymized analytics for public-health partners

The system uses **parallel agents**, **sequential orchestration**, and **LoopAgents with validation** to ensure correctness and reliability.

This allows the entire experience — education → booking → follow-up — to be automated, scalable, and reliable.


## **Architecture**

V-Access is built around an orchestrating agent: `vaccess_orchestrator_agent`.  
 This agent coordinates a family of specialists.

### **High-Level Architecture**


### **Agent Topology (Multi-Agent System)**

**1\. Education Specialist: `vaccine_info_agent`**

* LLM-powered.

* Provides explanations, myth-busting, eligibility guidance.

* Uses **context compaction** to keep conversations concise over long sessions.

**2\. Clinic Locator: `clinic_finder_agent`**

* Uses built-in Google Search \+ Maps API tools.

* Returns nearest valid clinics with hours, availability, address.

* Runs **in parallel** when user gives a zip code or location.

**3\. Appointment Scheduler: `appointment_agent`**

* Integrates with mock **OpenAPI booking tools**.

* Handles eligibility rules and appointment slot filtering.

* Implemented as a **LoopAgent** with a `ScheduleValidationChecker` to ensure all appointment details are valid before confirming.

**4\. Follow-up Specialist: `followup_agent`**

* Manages reminders using an ADK **long-running operation (pause/resume)**.

* Collects post-vaccination feedback.

* Stores notes in **Memory Bank** to persist across sessions.

**5\. Public Health Reporter: `analytics_agent`**

* Aggregates anonymized data for dashboards.

* Computes usage metrics, drop-off rates, and symptom reporting stats.

**6\. Central Coordinator: `vaccess_orchestrator_agent`**

* Sequentially orchestrates task phases.

* Delegates to sub-agents.

* Performs session & state tracking using `InMemorySessionService`.

* Handles context compaction.

* Ensures stability using fallback logic.


## **Essential Tools and Utilities**

### **Custom Tools**

* **`save_confirmation_to_file`**: Exports appointment confirmations.

* **`store_feedback`**: Saves user feedback into Memory Bank.

* **`clinic_directory_lookup`**: Calls a local or remote dataset to retrieve clinic metadata.

### **Built-in & External Tools**

* **Google Search / Maps**: Nearest clinic lookup.

* **OpenAPI Booking API**: Example vaccination scheduler.

* **Code Execution Tool**: Used by analytics agent for data aggregation.

### **Long-Running Operations**

Used by the **followup\_agent**:

* Pause agent execution until a specific reminder time.

* Resume automatically to send reminders or check-in messages.


## **Sessions & Memory**

We use:

* **`InMemorySessionService`** for tracking evolving interactions

* **Memory Bank** for longer-term data:

  * user’s preferred language

  * vaccination status

  * completed vs pending doses

  * reported symptoms

  * last consultation

Memory allows users to return days later and continue seamlessly.


## **Context Engineering**

Because healthcare conversations can become long:

* automatic **context compaction** is used before each sub-agent call

* redundant user utterances and stale tool outputs are pruned

* temporary tool contexts (e.g., one-time search results) are excluded unless needed


## **Observability: Logging, Tracing, Metrics**

This project logs:

* agent-to-agent calls

* tool invocations

* retries in LoopAgents

* appointment drop-off events

* reminder operations (pause/resume events)

Metrics collected:

* number of sessions completed

* successful appointment rates

* follow-up compliance

* symptom reporting frequency

* time spent in each agent stage

This mirrors the ADK’s recommended observability patterns.


## **Agent Evaluation**

The `eval/` directory includes:

* **Education accuracy tests**: ensure factual consistency

* **Clinic-finder correctness tests**: distance and relevance metrics

* **Appointment validation tests**: LoopAgent tests for valid slot selection

* **Follow-up test suite**: reminder timing and Memory Bank checks

* **End-to-end scenario test**: user → education → booking → follow-up


## **Conclusion**

V-Access demonstrates how multi-agent systems built with ADK can meaningfully support public-health needs. By breaking a complex healthcare workflow into modular, specialized agents, the system improves accuracy, trust, and scalability. It highlights how AI agents can be applied responsibly to solve socially important problems.


## **Value Statement**

V-Access significantly lowers the cognitive and logistical barriers that prevent people from receiving vaccines. It reduces the time required to understand eligibility, find clinics, schedule appointments, and complete follow-ups.

This agent system could meaningfully improve vaccine uptake in underserved areas — directly contributing to better community health and reduced healthcare disparities.

If more time were available, I would extend the system to:

* add a **trend-scanning research agent** to identify outbreaks or regional vaccine news using MCP servers

* integrate a multilingual text-to-speech agent for phone-based accessibility

* implement a cross-region analytics dashboard for public-health teams


## **Installation**

Python 3.11.3 recommended.  
 Install dependencies:

`pip install -r requirements.txt`


## **Running in ADK Web Mode**

`adk web`

## **Run Integration Tests**

`python -m tests.test_agent`


## **Project Structure**

`vaccess_agent/`  
`---- agent.py                      # Orchestrator definition`  
`---- sub_agents/`  
`-------- vaccine_info_agent.py     # Education`  
`-------- clinic_finder_agent.py    # Location search`  
`-------- appointment_agent.py      # Scheduling`  
`-------- followup_agent.py         # Reminders & check-ins`  
`-------- analytics_agent.py        # Reporting`  
`---- tools.py                      # Custom tools (save file, feedback, directory lookup)`  
`---- config.py                     # Model configs, tool configs`  

`eval/`  
`---- test_education.py`  
`---- test_clinics.py`  
`---- test_schedule.py`  
`---- test_followup.py`  
`---- test_e2e.py`  

`tests/`  
`---- test_agent.py                 # Base integration tests`


## **Workflow**

1. **Education Phase**  
    Orchestrator delegates to `vaccine_info_agent`. User learns about vaccine types, risks, eligibility.

2. **Clinic Search**  
    `clinic_finder_agent` runs using built-in search \+ map tools.

3. **Scheduling**  
    `appointment_agent` runs as a LoopAgent.  
    Repeats until `ScheduleValidationChecker` approves.

4. **Confirmation**  
    Orchestrator offers to save confirmation using custom tool.

5. **Follow-Up Reminders**  
    `followup_agent` schedules long-running reminders for second dose.

6. **Post-Vaccine Check-In**  
    Collects symptom feedback, stores into Memory Bank.

7. **Analytics**  
    `analytics_agent` aggregates anonymized metrics.


