from typing import Optional
import logging

from fastapi import APIRouter, HTTPException, WebSocket
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.models.telephony import (
    BaseCallConfig,
    TwilioCallConfig,
    VonageCallConfig,
)
from vocode.streaming.synthesizer.factory import SynthesizerFactory
from vocode.streaming.telephony.config_manager.base_config_manager import (
    BaseConfigManager,
)

from vocode.streaming.telephony.conversation.call import Call
from vocode.streaming.telephony.conversation.twilio_call import TwilioCall
from vocode.streaming.telephony.conversation.vonage_call import VonageCall
from vocode.streaming.transcriber.factory import TranscriberFactory
from vocode.streaming.utils.base_router import BaseRouter
from vocode.streaming.utils.events_manager import EventsManager

from opentelemetry import trace
import traceback
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace import TracerProvider 
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vocode.streaming.telephony.server.utils import SpanLogHandler, DatabaseExporter


class CallsRouter(BaseRouter):
    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        transcriber_factory: TranscriberFactory = TranscriberFactory(),
        agent_factory: AgentFactory = AgentFactory(),
        synthesizer_factory: SynthesizerFactory = SynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        self.base_url = base_url
        self.config_manager = config_manager
        self.transcriber_factory = transcriber_factory
        self.agent_factory = agent_factory
        self.synthesizer_factory = synthesizer_factory
        self.events_manager = events_manager
        self.logger = logger or logging.getLogger(__name__)
        self.router = APIRouter()
        self.router.websocket("/connect_call/{id}")(self.connect_call)
        self.handler_set = False

    def _from_call_config(
        self,
        base_url: str,
        call_config: BaseCallConfig,
        config_manager: BaseConfigManager,
        conversation_id: str,
        logger: logging.Logger,
        transcriber_factory: TranscriberFactory = TranscriberFactory(),
        agent_factory: AgentFactory = AgentFactory(),
        synthesizer_factory: SynthesizerFactory = SynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
    ):
        if isinstance(call_config, TwilioCallConfig):
            return TwilioCall(
                to_phone=call_config.to_phone,
                from_phone=call_config.from_phone,
                base_url=base_url,
                logger=logger,
                config_manager=config_manager,
                agent_config=call_config.agent_config,
                transcriber_config=call_config.transcriber_config,
                synthesizer_config=call_config.synthesizer_config,
                twilio_config=call_config.twilio_config,
                twilio_sid=call_config.twilio_sid,
                conversation_id=conversation_id,
                transcriber_factory=transcriber_factory,
                agent_factory=agent_factory,
                synthesizer_factory=synthesizer_factory,
                events_manager=events_manager,
            )
        elif isinstance(call_config, VonageCallConfig):
            return VonageCall(
                to_phone=call_config.to_phone,
                from_phone=call_config.from_phone,
                base_url=base_url,
                logger=logger,
                config_manager=config_manager,
                agent_config=call_config.agent_config,
                transcriber_config=call_config.transcriber_config,
                synthesizer_config=call_config.synthesizer_config,
                vonage_config=call_config.vonage_config,
                vonage_uuid=call_config.vonage_uuid,
                conversation_id=conversation_id,
                transcriber_factory=transcriber_factory,
                agent_factory=agent_factory,
                synthesizer_factory=synthesizer_factory,
                events_manager=events_manager,
                output_to_speaker=call_config.output_to_speaker,
            )
        else:
            raise ValueError(f"Unknown call config type {call_config.type}")

    async def connect_call(self, websocket: WebSocket, id: str):
        span_exporter = InMemorySpanExporter()
        database_exporter = DatabaseExporter(id, self.logger)
        span_processor = SimpleSpanProcessor(span_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("connect_call") as conversation:
            if not self.handler_set: self.logger.addHandler(SpanLogHandler())
            self.handler_set = True
            self.logger.debug("waiting for websocket to accept")
            await websocket.accept()
            self.logger.debug("Phone WS connection opened for chat {}".format(id))
            self.logger.debug("Kwal vocode being used")
            call_config = await self.config_manager.get_config(id)
            if not call_config:
                raise HTTPException(status_code=400, detail="No active phone call")
            try:
                call = self._from_call_config(
                    base_url=self.base_url,
                    call_config=call_config,
                    config_manager=self.config_manager,
                    conversation_id=id,
                    transcriber_factory=self.transcriber_factory,
                    agent_factory=self.agent_factory,
                    synthesizer_factory=self.synthesizer_factory,
                    events_manager=self.events_manager,
                    logger=self.logger,
                )
                self.logger.debug("testing here3")

                await call.attach_ws_and_start(websocket)
                self.logger.debug("testing here4")
            except Exception as e:
                self.logger.error(f"Error {e}, Trace: {traceback.format_exc()}")
            self.logger.debug("Phone WS connection closed for chat {}".format(id))
        child_spans = span_exporter.get_finished_spans()

        try:
            await database_exporter.export(child_spans, from_phone=call_config.from_phone, to_phone=call_config.to_phone)
        except Exception as e:
            self.logger.error(f"Error {e}, Trace: {traceback.format_exc()}")

    def get_router(self) -> APIRouter:
        return self.router
