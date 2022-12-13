# Changelog

All notable changes to this project will be documented in this file. See
[Conventional Commits](https://conventionalcommits.org) for commit guidelines.

## [1.10.0](https://github.com/emma-simbot/policy/compare/v1.9.0...v1.10.0) (2022-12-13)


### Features

* Add low level action predictor with raw text match ([#644](https://github.com/emma-simbot/policy/issues/644)) ([a8ad897](https://github.com/emma-simbot/policy/commit/a8ad897a28cf76f5150d27eb40766c750887873f))

## [1.9.0](https://github.com/emma-simbot/policy/compare/v1.8.0...v1.9.0) (2022-12-05)


### Features

* Add new act types (low level and search) to NLU ([#630](https://github.com/emma-simbot/policy/issues/630)) ([97406fc](https://github.com/emma-simbot/policy/commit/97406fc26d3776d90af41519d5c2720c7f6406a2))

## [1.8.0](https://github.com/emma-simbot/policy/compare/v1.7.0...v1.8.0) (2022-11-30)


### Features

* Create find endpoint ([#641](https://github.com/emma-simbot/policy/issues/641)) ([5737b55](https://github.com/emma-simbot/policy/commit/5737b558a9984f73bbeedf30a32a69b155d50b6d))

## [1.7.0](https://github.com/emma-simbot/policy/compare/v1.6.0...v1.7.0) (2022-11-30)


### Features

* Keep only target frame for each action ([#640](https://github.com/emma-simbot/policy/issues/640)) ([96a30a4](https://github.com/emma-simbot/policy/commit/96a30a40546a2d4fd3101369550c0422d3b4d0d1))

## [1.6.0](https://github.com/emma-simbot/policy/compare/v1.5.2...v1.6.0) (2022-11-24)


### Features

* Add distributed weighted sampler ([#634](https://github.com/emma-simbot/policy/issues/634)) ([3f0f82b](https://github.com/emma-simbot/policy/commit/3f0f82baad94d4037b95ea824cb7beee9e0e594b))


### Bug Fixes

* test case of all zero weights ([#636](https://github.com/emma-simbot/policy/issues/636)) ([2313f20](https://github.com/emma-simbot/policy/commit/2313f20a9491d2cc168620cfebf7581d1fa35ec8))

## [1.5.2](https://github.com/emma-simbot/policy/compare/v1.5.1...v1.5.2) (2022-11-17)


### Bug Fixes

* **hack/simbot-nlu:** comment out everything to do with the `SimBotClarificationTypes` ([#628](https://github.com/emma-simbot/policy/issues/628)) ([68a2176](https://github.com/emma-simbot/policy/commit/68a21763242feb91af308e798648559cd410e001))

## [1.5.1](https://github.com/emma-simbot/policy/compare/v1.5.0...v1.5.1) (2022-11-11)


### Bug Fixes

* Fix bugs in simbot datasets ([#625](https://github.com/emma-simbot/policy/issues/625)) ([6672154](https://github.com/emma-simbot/policy/commit/667215445491d69503a6cedb7f4200d82539310d))

## [1.5.0](https://github.com/emma-simbot/policy/compare/v1.4.1...v1.5.0) (2022-11-10)


### Features

* Paraphrase simbot actions ([#623](https://github.com/emma-simbot/policy/issues/623)) ([af31c1a](https://github.com/emma-simbot/policy/commit/af31c1a8a362d96d4e86720d7429c304e6404cc8))

## [1.4.1](https://github.com/emma-simbot/policy/compare/v1.4.0...v1.4.1) (2022-11-03)


### Bug Fixes

* Fix bug from NLU imports ([#621](https://github.com/emma-simbot/policy/issues/621)) ([26cbdca](https://github.com/emma-simbot/policy/commit/26cbdca4618e0371471ae4e7ede67427b2f67d6b))

## [1.4.0](https://github.com/emma-simbot/policy/compare/v1.3.0...v1.4.0) (2022-11-03)


### Features

* Update NLU and Action Prediction with synthetic ambiguous go-to instructions ([#614](https://github.com/emma-simbot/policy/issues/614)) ([85381fe](https://github.com/emma-simbot/policy/commit/85381fe779751cf32abb0c651f9f88d384905f9f))

## [1.3.0](https://github.com/emma-simbot/policy/compare/v1.2.0...v1.3.0) (2022-11-02)


### Features

* Decode images ([#618](https://github.com/emma-simbot/policy/issues/618)) ([a02f41b](https://github.com/emma-simbot/policy/commit/a02f41bdcf1eee4e09a5827609e17f9123064466))

## [1.2.0](https://github.com/emma-simbot/policy/compare/v1.1.2...v1.2.0) (2022-11-01)


### Features

* Reduce false positives for clarification ([#619](https://github.com/emma-simbot/policy/issues/619)) ([ad88eb7](https://github.com/emma-simbot/policy/commit/ad88eb72a85a0fdd1393c7dc9934dcf72e0497b4))

## [1.1.2](https://github.com/emma-simbot/policy/compare/v1.1.1...v1.1.2) (2022-10-31)


### Bug Fixes

* Fix bug in input text preparation ([#620](https://github.com/emma-simbot/policy/issues/620)) ([c83768c](https://github.com/emma-simbot/policy/commit/c83768c16f2c824207efd60b9deef9bbdbd4768c))

## [1.1.1](https://github.com/emma-simbot/policy/compare/v1.1.0...v1.1.1) (2022-10-29)


### Bug Fixes

* re-add fastapi as a web dependency and update the dockerfile ([cf2c778](https://github.com/emma-simbot/policy/commit/cf2c77806cee2308180bd409afcae1810bb20e2e))

## [1.1.0](https://github.com/emma-simbot/policy/compare/v1.0.0...v1.1.0) (2022-10-28)


### Features

* Replace QA format with speaker tokens ([#616](https://github.com/emma-simbot/policy/issues/616)) ([895cb8d](https://github.com/emma-simbot/policy/commit/895cb8d65c03459b0581b276f0a7d80aa4756747))

## 1.0.0 (2022-10-27)


### Features

* Ban generating past frames ([#604](https://github.com/emma-simbot/policy/issues/604)) ([a356c6b](https://github.com/emma-simbot/policy/commit/a356c6bbb51ba4f472fc077dd135f9cb9cca419c))
* be able to change the number of target tokens per decoding step from the args ([#395](https://github.com/emma-simbot/policy/issues/395)) ([0782e71](https://github.com/emma-simbot/policy/commit/0782e719cf02276c0a6b359d503f85ebbd2e38f0))
* fix the image comparison to stop being false positive ([#378](https://github.com/emma-simbot/policy/issues/378)) ([558e1cb](https://github.com/emma-simbot/policy/commit/558e1cb903cd29f68f5400d9d2a6037dddfa6574))
* Subsample look and goto table/desk ([#607](https://github.com/emma-simbot/policy/issues/607)) ([09344bc](https://github.com/emma-simbot/policy/commit/09344bc444ab6bff18c126984ab9ee7910f7d957))


### Bug Fixes

* Add endpoint in run_model ([#615](https://github.com/emma-simbot/policy/issues/615)) ([53fa772](https://github.com/emma-simbot/policy/commit/53fa772e5135699f303ce005be60653fbb74efeb))
* bounding boxes ([#382](https://github.com/emma-simbot/policy/issues/382)) ([a834f5d](https://github.com/emma-simbot/policy/commit/a834f5dac35a70588ac3376283bd8938591e83df))
* dataloader types in emma pretrain datamodule ([#137](https://github.com/emma-simbot/policy/issues/137)) ([b317198](https://github.com/emma-simbot/policy/commit/b317198f6c0f831cec4098c4cfc266b26399347f))
* empty trajectory or action delimiter only ([#398](https://github.com/emma-simbot/policy/issues/398)) ([028cb94](https://github.com/emma-simbot/policy/commit/028cb9450fef071ea648bfcd3616adde1bfd7577))
* input encoding needs to be the same as the training ([#380](https://github.com/emma-simbot/policy/issues/380)) ([094b984](https://github.com/emma-simbot/policy/commit/094b98474b4c2c06162fa8509aeac1f677dfbf9f))
* original history length set to minumum ([#386](https://github.com/emma-simbot/policy/issues/386)) ([659721d](https://github.com/emma-simbot/policy/commit/659721d46948112936321d4dcaa66f6f50734c17))
* pick a random object index from the ones in the current frame ([#393](https://github.com/emma-simbot/policy/issues/393)) ([565beb5](https://github.com/emma-simbot/policy/commit/565beb526a1d1d02db35c4bee966f9ff3531c0fe))
* set the original history length to be max at the max frames ([#383](https://github.com/emma-simbot/policy/issues/383)) ([39df383](https://github.com/emma-simbot/policy/commit/39df3838f3dc029116da17de4ed3f9f8f8968cae))
* update the previous frame AFTER the comparison is made ([#381](https://github.com/emma-simbot/policy/issues/381)) ([6e8f979](https://github.com/emma-simbot/policy/commit/6e8f979d667e2cf8d80e63ab03e44d9eeb505cdb))


### Reverts

* Revert "fix: remove the remaining load ref coco images function" ([2d35446](https://github.com/emma-simbot/policy/commit/2d354467f20d97bc74fde9fa8e43e9d89892627a))
* Revert "Run the tests CI faster (#211)" ([461edff](https://github.com/emma-simbot/policy/commit/461edffe3255d966ac9c166893e8b06250dae72d)), closes [#211](https://github.com/emma-simbot/policy/issues/211)
