## Running `Hasura Console - Local Development`

**Error:** 
> Executing task: docker compose --env-file ./hasura/.env.local up -d <  
>   
> unknown flag: --env-file  
> See 'docker --help'.  

**Fix:**  
Change command from `docker compose` to `docker-compose`

---

**Error:**
> FATA[0002] version check: failed to get version from server: failed making version api call: Get "http://127.0.0.1:8080/v1/version": read tcp 127.0.0.1:60550->127.0.0.1:8080: read: connection reset by peer   
> The terminal process "bash '-c', 'hasura console --envfile .env.local'" terminated with exit code: 1.  

**Fix:** 
Disconnect from VPN

---
**Error:**
> FATA[0000] version check: failed to get version from server: failed making version api call: Get "http://127.0.0.1:8080/v1/version": read tcp 127.0.0.1:33310->127.0.0.1:8080: read: connection reset by peer   
> The terminal process "bash '-c', 'hasura console --envfile .env.local'" terminated with exit code: 1.  

**Fix:**  
Wait a second and restart the task - likely caused by the race condition mentioned in the README
For me I sometimes need to do three attempts because the second attempt only started up another docker container and then threw the same error as above

## Running `Actions Service - Local Development`

**Error:**
> Executing task: npm run dev <  
>  
> clowdr-actions@1.0.0 dev  
> npm run build-shared && nodenv -E ./.env -e "npm run dev-stage2"  
> clowdr-actions@1.0.0 build-shared  
> npm run --prefix=../../ --workspace=shared build  
>  
> npm ERR! missing script: build  

**Fix:**  
change the following line in `actions/package.json`:
- from: `"build-shared": "npm run --prefix=../../ --workspace=shared build",`  
- to: `"build-shared": "npm run --prefix=../../shared build",`

--- 

**Error:**
> clowdr@1.0.0 run-services-actions  
> npx foreman start --procfile ./Procfile -p 3001  
>   
> 12:44:12 web.1   |  node:assert:402  
> 12:44:12 web.1   |      throw err;  
> 12:44:12 web.1   |      ^  
> 12:44:12 web.1   |  AssertionError [ERR_ASSERTION]: CORS_ORIGIN env var not provided.  
> 12:44:12 web.1   |      at Object.<anonymous> (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/actions/build/router/companion.js:19:17)  
> 12:44:12 web.1   |      at Module._compile (node:internal/modules/cjs/loader:1092:14)  
> 12:44:12 web.1   |      at Object.Module._extensions..js (node:internal/modules/cjs/loader:1121:10)  
> 12:44:12 web.1   |      at Module.load (node:internal/modules/cjs/loader:972:32)  
> 12:44:12 web.1   |      at Function.Module._load (node:internal/modules/cjs/loader:813:14)  
> 12:44:12 web.1   |      at Module.require (node:internal/modules/cjs/loader:996:19)  
> 12:44:12 web.1   |      at require (node:internal/modules/cjs/helpers:92:18)  
> 12:44:12 web.1   |      at Object.<anonymous> (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/actions/build/server.js:57:19)  
> 12:44:12 web.1   |      at Module._compile (node:internal/modules/cjs/loader:1092:14)  
> 12:44:12 web.1   |      at Object.Module._extensions..js (node:internal/modules/cjs/loader:1121:10) {  
> 12:44:12 web.1   |    generatedMessage: false,  
> 12:44:12 web.1   |    code: 'ERR_ASSERTION',  
> 12:44:12 web.1   |    actual: undefined,  
> 12:44:12 web.1   |    expected: true,  
> 12:44:12 web.1   |    operator: '=='  
> 12:44:12 web.1   |  }  
> [DONE] Killing all processes with signal  SIGINT  
> 12:44:12 web.1   Exited with exit code null  
 
**Fix:**  
No fix found until now. Further debugging needed.

## Running `Playout Service - Local Development`
**Error:**
> /home/lasse/Programming/Uni/Honours_Project/clowdr/services/playout/src/aws/aws.module.ts:72  
>         assert(cloudFormationNotificationsTopicArn, "Missing AWS_CLOUDFORMATION_NOTIFICATIONS_TOPIC_ARN");  
>               ^  
> AssertionError [ERR_ASSERTION]: Missing AWS_CLOUDFORMATION_NOTIFICATIONS_TOPIC_ARN  
>     at AwsModule.onModuleInit (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/playout/src/aws/aws.module.ts:72:15)  
>     at Object.callModuleInitHook (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/playout/node_modules/@nestjs/core/hooks/on-module-init.hook.js:51:35)  
>     at processTicksAndRejections (node:internal/process/task_queues:94:5)  
>     at NestApplication.callInitHook (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/playout/node_modules/@nestjs/core/nest-application-context.js:166:13)  
>     at NestApplication.init (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/playout/node_modules/@nestjs/core/nest-application.js:93:9)  
>     at NestApplication.listen (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/playout/node_modules/@nestjs/core/nest-application.js:147:33)  
>     at bootstrap (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/playout/src/main.ts:13:5)  

**Fix:**  
No fix found - seems to have to do with missing AWS setup.


## Running `Real-time Service - Local Development`

**Error:**
> Executing task: npm run dev <  
>   
>   
> clowdr-presence@1.0.0 dev  
> npm run build-shared && node-env-run -E ./.env -e "npm run dev-stage2"  
>   
>   
> clowdr-presence@1.0.0 build-shared  
> npm run --prefix=../../ --workspace=shared build  
>   
> npm ERR! missing script: build 

**Fix:**  
change the following line in `realtime/package.json`:
- from: `"build-shared": "npm run --prefix=../../ --workspace=shared build",`  
- to: `"build-shared": "npm run --prefix=../../shared build",`

---

**Error:**
> 11:52:21 chatReactionsWritebackWorker.1      |    cause: Error: Handshake terminated by server: 403 (ACCESS-REFUSED) with message "ACCESS_REFUSED - Login was refused using authentication mechanism PLAIN. For details see the broker logfile."  
> 11:52:21 chatReactionsWritebackWorker.1      |        at afterStartOk (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/realtime/node_modules/amqplib/lib/connection.js:220:12)  
> 11:52:21 chatReactionsWritebackWorker.1      |        at /home/lasse/Programming/Uni/Honours_Project/clowdr/services/realtime/node_modules/amqplib/lib/connection.js:160:12  
> 11:52:21 chatReactionsWritebackWorker.1      |        at Socket.recv (/home/lasse/Programming/Uni/Honours_Project/clowdr/services/realtime/node_modules/amqplib/lib/connection.js:499:12)  
> 11:52:21 chatReactionsWritebackWorker.1      |        at Object.onceWrapper (node:events:475:28)  
> 11:52:21 chatReactionsWritebackWorker.1      |        at Socket.emit (node:events:369:20)  
> 11:52:21 chatReactionsWritebackWorker.1      |        at emitReadable_ (node:internal/streams/readable:574:12)  
> 11:52:21 chatReactionsWritebackWorker.1      |        at processTicksAndRejections (node:internal/process/task_queues:80:21),  
> 11:52:21 chatReactionsWritebackWorker.1      |    isOperational: true  
> 11:52:21 chatReactionsWritebackWorker.1      |  }  
> 11:52:21 chatReactionsWritebackWorker.1      Exited with exit code SIGINT 

+ similar error for the other chat workers  

**Fix:**  
Changing the config variables in `realtime/.env` and `playout/.env` as follows:
- `RABBITMQ_USERNAME=guest`
- `RABBITMQ_PASSWORD=guest`

**Remark:**
- It seems like ones you added a `guest:guest` user through the admin interface of RabbitMQ, all other combinations of users work as well, e.g. `services/realtime:test`
  - That's unexpected... 
--- 


## Running `Frontend - Local Development`
**Error:**
> Executing task: npm start <  
  
  
> start  
> npm run build-shared && snowpack dev  
  
  
> build-shared  
> npm run --prefix=../ --workspace=shared build  
  
> npm ERR! missing script: build  
 
**Fix:**  
change the following line in `frontend/package.json`:
- from: `"build-shared": "npm run --prefix=../ --workspace=shared build",`  
- to: `"build-shared": "npm run --prefix=../shared build",`

---
**Error:**  
Frontend loads but displays the following error:
> Unhandled Runtime Error  
>   
> SyntaxError: expected expression, got '='  
>  
> Source  
>  http://localhost:3000/_dist_/aspects/Conference/Manage/ManageGroups.js [:270:19]   

**Fix:**  
No fix found yet, further debugging required
