# Setup Notes
_Things I realised during the local setup of clowdr_

## General
- What does BCP mean: "Best Current Practice" ?
- for me it wasn't clear that you should only do the pre-requisites section on each subpage
  - I started with the setup of the different parts straight away which led to a bit of confusion later on
    - I naively thought that the setup was part of the pre-requisites for later setup steps
  - maybe there is a way to make it clearer that the pre-requisite links are
    only supposed to link the top section of each subpage


## Steps
#### General Setup (main README)
**Step 3**
- building the slate-transcript-editor creates the dist folder but also creates a .diff file
  - should this .diff file (slate-transcript-editor.diff) be added to the .gitignore or should it just be ignored?
- there are also some changes in the package-lock.json after building 
  - can this just be ignored? 

---
#### Specific Setups (sub README pages)
**Hasura Setup**
- change docker command to docker-compose

- if you are connected to a **VPN** you need to disconnect to connect to local services
  - or reroute LAN traffic to local gateway

**Clowdr: Actions Service**
- which video codec format should I use for my Vonage Video API Account? (VP8 or H.264)
  - from the code I assume H264 (clowdr/services/playout/src/channel-stack/channel-stack/channelStack.ts)
- GRAPHQL_API_SECURE_PROTOCOLS says default _true_ but it's _false_
- Order Env-variables according to env file or alphabetically? 
  - e.g. SENDGRID_API_KEY and SENDGRID_SENDER are in a different position than in the env file
- Quick Setup: what about the GOOGLE CLOUD and SENDGRID variables? 
  - should they be `XXXX` as well? 
- Do I need to run `Actions service - GraphQL Codegen` for quick setup as well? 

- **Error I ran into:**
  - it seems like some files depend on generated code, namely "generated/graphql"
  - these files seem to be called during the `Actions service - GraphQL Codegen` task
    - since the code hasn't been generated an error is thrown
  - **FIX**: Restart whole setup and run task again 
    - docker-compose down
    - then run `Hasura Console -- Local Development task` (which also runs docker-compose)
    - then run `Actions service - GraphQL Codegen`

**Clowdr: Playout Service**
- example.env: 
  - EVENT_SECRET would be better as `XXXXX` instead of `XXXXY`

**Expose local services at a public URL**
- _"Copy the auth URL (http://<hasura-domain>/v1/graphql) into the HASURA_URL Auth0 Rule Configuration as shown in step 5."_
  - which step 5? 
    - **STEP 5 of the Auth0 setup**

**PKTRIOT**
- If you only have the FREE pktriot plan `pktriot start` won't work because you only have one port per tunnel and the setup (`pktriot.json`) defines 3 ports


- If you try to start pktriot with only prior steps completed you get the following error:
  `{"status":false,"errror":"invalid accept value in request"}`
  - **FIXED** There was a mistake in the `pktriot.json`


**Auth0 Setup**
- For all configured URLs the trailing comma in the last line, needed to be removed for Auth0 to accept the input
  - e.g.: 
    > http://localhost:3000/auth0/,  
    > http://localhost:3000/auth0/logged-in,  
    > http://localhost:3000/auth0/email-verification/result **,** <- this comma

- `Rule` Tab is now in the sidebar under `Auth Pipeline/Rules`
- There are two step 1s under `4. Create Rules`

## How to run Local Setup after initialisation
1) Run `Hasura Console -- Local Development` task
- _Your browser should have opened a tab to the Hasura console_

2) Open http://localhost:15672 in a web browser and log in with the default username and password, both admin.
  - Now, create the user that the realtime service will use to access RabbitMQ. Go to the Admin tab and add a new user with username services/realtime and password 1234. Click the username of the newly-created user, and then click the Set Permission button. This gives the user unrestricted read/write access.


## TODO
- On the hasura url that is published via pktriot it says:
  -> resource does not exist
