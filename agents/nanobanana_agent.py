"""
Nanobanana Agent: Gemini Flash-powered image generation agent.

Specializes in creating scientific illustrations, cover images,
and graphical abstracts for research papers.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from google import genai
from google.genai import types
from PIL import Image

from utils.logging import get_logger

logger = get_logger(__name__)


class NanobananaAgent:
    """
    Expert agent for creating scientific illustrations using Gemini Flash.

    Capabilities:
    - Generate article cover images
    - Create graphical abstracts
    - Process and enhance existing images
    - Add annotations to figures
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-image",  # Nano Banana model
        output_dir: str = "./plots",
    ):
        """
        Initialize nanobanana agent.

        Args:
            api_key: Google Gemini API key
            model: Gemini model name (gemini-2.5-flash-image or gemini-3-pro-image-preview)
            output_dir: Directory to save generated images
        """
        self.api_key = api_key
        self.model_name = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure Gemini
        self.client = genai.Client(api_key=api_key)

        logger.info(f"Initialized nanobanana agent with model {model}")

    async def create_cover_image(
        self,
        prompt: str,
        style: str = "scientific",
        size: Tuple[int, int] = (1200, 800),
    ) -> str:
        """
        Generate article cover image using Nano Banana.

        Args:
            prompt: Description of the image to generate
            style: Visual style ('scientific', 'abstract', 'technical')
            size: Image size (width, height)

        Returns:
            Path to generated image file
        """
        logger.info(f"Generating cover image: {prompt[:50]}...")

        # Enhance prompt with style guidance
        enhanced_prompt = self._enhance_prompt(prompt, style, "cover")

        # Generate filename
        filename = f"cover_{uuid.uuid4().hex[:8]}.png"
        output_path = self.output_dir / filename

        try:
            # Generate image using Nano Banana
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=enhanced_prompt,
            )

            # Extract image from response
            image_saved = False
            for part in response.parts:
                if part.inline_data:
                    image = part.as_image()
                    image.save(output_path, format="PNG")
                    image_saved = True
                    logger.info(f"✓ Generated cover image: {output_path}")
                    break

            if not image_saved:
                logger.warning(f"No image generated. Response: {response.text if response.candidates else 'No response'}")
                raise ValueError("Nano Banana did not return an image. The model may not support image generation or the prompt was rejected.")

            return str(output_path.absolute())

        except Exception as e:
            logger.error(f"✗ Error generating cover image: {e}")
            raise

    async def create_abstract_figure(
        self,
        description: str,
        style: str = "clean",
    ) -> str:
        """
        Create graphical abstract for scientific paper using Nano Banana.

        Args:
            description: Description of the abstract content
            style: Visual style preference

        Returns:
            Path to generated figure file
        """
        logger.info(f"Generating abstract figure: {description[:50]}...")

        enhanced_prompt = self._enhance_prompt(description, style, "abstract")
        filename = f"abstract_{uuid.uuid4().hex[:8]}.png"
        output_path = self.output_dir / filename

        try:
            # Generate image using Nano Banana
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=enhanced_prompt,
            )

            # Extract image from response
            image_saved = False
            for part in response.parts:
                if part.inline_data:
                    image = part.as_image()
                    image.save(output_path, format="PNG")
                    image_saved = True
                    logger.info(f"✓ Generated abstract figure: {output_path}")
                    break

            if not image_saved:
                logger.warning(f"No image generated. Response: {response.text if response.candidates else 'No response'}")
                raise ValueError("Nano Banana did not return an image.")

            return str(output_path.absolute())

        except Exception as e:
            logger.error(f"✗ Error generating abstract figure: {e}")
            raise

    async def process_image(
        self,
        image_path: str,
        instruction: str,
    ) -> str:
        """
        Process existing image with instructions.

        Note: This uses Gemini's vision capabilities for analysis, not Nano Banana.
        For actual image editing, consider using dedicated image editing tools.

        Args:
            image_path: Path to input image
            instruction: Processing instructions

        Returns:
            Path to analysis/description file
        """
        logger.info(f"Processing image: {image_path}")

        input_path = Path(image_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        filename = f"processed_{uuid.uuid4().hex[:8]}.txt"
        output_path = self.output_dir / filename

        try:
            # Use Gemini's vision capabilities to analyze the image
            # Note: Using a vision model, not Nano Banana, for image analysis
            with open(input_path, "rb") as f:
                image_data = f.read()

            # Create multimodal request with vision model
            vision_model = "gemini-2.0-flash-exp"  # Use vision model for analysis
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=vision_model,
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type="image/png"),
                    types.Part.from_text(f"Analyze this image and provide detailed suggestions for: {instruction}"),
                ],
            )

            analysis = response.text if response.candidates else "No analysis generated"

            # Write analysis
            with open(output_path, "w") as f:
                f.write(f"Image Analysis\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Input Image: {image_path}\n")
                f.write(f"Instruction: {instruction}\n\n")
                f.write(f"Analysis:\n")
                f.write(f"{analysis}\n")

            logger.info(f"✓ Analyzed image: {output_path}")
            return str(output_path.absolute())

        except Exception as e:
            logger.error(f"✗ Error processing image: {e}")
            raise

    async def _poll_generation_status(
        self,
        generation_url: str,
        max_attempts: int = 30,
        poll_interval: int = 2,
    ) -> bytes:
        """
        Poll async generation URL until complete.

        Args:
            generation_url: URL to poll for status
            max_attempts: Maximum polling attempts
            poll_interval: Seconds between polls

        Returns:
            Generated image data

        Raises:
            TimeoutError: If generation times out
        """
        logger.debug(f"Polling generation status: {generation_url}")

        for attempt in range(max_attempts):
            try:
                # This would poll an actual async generation endpoint
                # Placeholder implementation
                await asyncio.sleep(poll_interval)

                # Check if complete (placeholder logic)
                # In real implementation, check the actual status endpoint
                logger.debug(f"Poll attempt {attempt + 1}/{max_attempts}")

                # Simulate completion after a few attempts
                if attempt >= 3:
                    logger.info("✓ Generation complete")
                    return b"<placeholder image data>"

            except Exception as e:
                logger.warning(f"Poll attempt {attempt + 1} failed: {e}")

        raise TimeoutError(f"Generation timed out after {max_attempts * poll_interval}s")

    def _enhance_prompt(self, prompt: str, style: str, image_type: str) -> str:
        """
        Enhance user prompt with style and context guidance.

        Args:
            prompt: User's original prompt
            style: Visual style
            image_type: Type of image (cover, abstract, etc.)

        Returns:
            Enhanced prompt
        """
        style_guides = {
            "scientific": "professional, clean, high-contrast, publication-quality",
            "abstract": "minimal, conceptual, symbolic, modern",
            "technical": "detailed, precise, informative, schematic",
            "clean": "simple, clear, uncluttered, elegant",
        }

        style_desc = style_guides.get(style, "professional")

        if image_type == "cover":
            template = f"Create a {style_desc} scientific journal cover image depicting: {prompt}"
        elif image_type == "abstract":
            template = f"Design a {style_desc} graphical abstract showing: {prompt}"
        else:
            template = f"Generate a {style_desc} scientific illustration of: {prompt}"

        return template

    async def handle(self, tool_call: dict) -> dict:
        """
        Handle tool call from orchestrator.

        Args:
            tool_call: Tool call dict with name and arguments

        Returns:
            Tool result dict
        """
        tool_name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        try:
            if tool_name == "nanobanana_create_cover_image":
                result = await self.create_cover_image(
                    prompt=args.get("prompt"),
                    style=args.get("style", "scientific"),
                )
                return {"success": True, "file_path": result}

            elif tool_name == "nanobanana_create_abstract_figure":
                result = await self.create_abstract_figure(
                    description=args.get("description"),
                    style=args.get("style", "clean"),
                )
                return {"success": True, "file_path": result}

            elif tool_name == "nanobanana_process_image":
                result = await self.process_image(
                    image_path=args.get("image_path"),
                    instruction=args.get("instruction"),
                )
                return {"success": True, "file_path": result}

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"nanobanana tool call failed: {e}")
            return {"success": False, "error": str(e)}
