# Setup Notes
_Things I realised during the local setup of clowdr_

## General
- What does BCP mean: "Best Current Practice" ?
- for me it wasn't clear that you should only do the pre-requisites section on each subpage
  - I started with the setup of the different parts straight away which led to a bit of confusion later on
    - I naively thought that the setup was part of the pre-requisites for later setup steps
  - maybe there is a way to make it clearer that the pre-requisite links are
    only supposed to link the top section of each subpage

###Steps

**Step 3**
- building the slate-transcript-editor creates the dist folder but also creates a .diff file
  - should this .diff file (slate-transcript-editor.diff) be added to the .gitignore or should it just be ignored?
- there are also some changes in the package-lock.json after building 
  - can this just be ignored? 


**Hasura Setup**
- change docker command to docker-compose

- if you are connected to a vpn you need to disconnect to connect to local services
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


