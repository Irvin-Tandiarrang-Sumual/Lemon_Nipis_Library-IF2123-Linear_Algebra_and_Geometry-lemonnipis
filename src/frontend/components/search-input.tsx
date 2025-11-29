"use client";

import { useRef, useState } from "react";
import { Input } from "@heroui/input";
import { SearchIcon, CameraIcon, CloseIcon } from "@/components/icons";

export const CustomSearchInput = ({ className }: { className?: string }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const handleAutoSearchImage = (file: File) => {
    alert(`[Auto Search] Mencari gambar: ${file.name}`);
  };

  const handleTextSearch = (query: string) => {
    if (!query.trim()) return;
    alert(`[Text Search] Mencari teks: "${query}"`);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleTextSearch(e.currentTarget.value);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      handleAutoSearchImage(file);
      event.target.value = "";
    }
  };

  const handleCameraClick = () => {
    fileInputRef.current?.click();
  };

  const handleClearImage = () => {
    setSelectedImage(null);
  };

  return (
    <div className={className}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="image/*"
      />

      <Input
        aria-label="Search"
        classNames={{
          inputWrapper: "bg-default-100",
          input: "text-sm",
          base: "w-full h-10",
        }}
        onKeyDown={handleKeyDown}
        startContent={
          selectedImage ? (
            <div className="flex items-center gap-2 mr-1">
              <div className="relative w-8 h-8 rounded overflow-hidden border border-default-300">
                <img src={selectedImage} alt="Preview" className="w-full h-full object-cover" />
              </div>
              <button
                onClick={handleClearImage}
                className="text-default-400 hover:text-danger p-0.5 rounded-full hover:bg-default-200 transition-colors"
                type="button"
              >
                <CloseIcon className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <SearchIcon className="text-base text-default-400 pointer-events-none flex-shrink-0" />
          )
        }
        endContent={
          <button
            className="focus:outline-none hover:opacity-70 text-default-400 hover:text-primary transition-colors"
            type="button"
            onClick={handleCameraClick}
            title="Upload Image"
          >
            <CameraIcon className="flex-shrink-0" />
          </button>
        }
        labelPlacement="outside"
        placeholder={selectedImage ? "Add description..." : "Search text or import image..."}
        type="search"
      />
    </div>
  );
};